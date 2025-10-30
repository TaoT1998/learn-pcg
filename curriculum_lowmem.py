#!/usr/bin/env python3
"""
Usage:
  - Basic training:
    python curriculum_lowmem.py --config configs/curriculum.yaml
  
  - Fine-tuning from pretrained model:
    python curriculum_lowmem.py --config configs/curriculum.yaml \
      --pretrained_path path/to/multi_curriculum_model_...pt
  
  - With vocabulary expansion:
    python curriculum_lowmem.py --config configs/curriculum.yaml \
      --pretrained_path path/to/model_vocab512.pt --vocab_size 1024
"""

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import time
import sys
import math
import numpy as np
import pandas as pd
import platform
import psutil
import os
import argparse
import yaml
from socket import gethostname

# Import W&B utilities
from utils.wandb_utils import (
    init_wandb, log_training_metrics, log_correctness_plot, 
    finish_wandb, add_wandb_args, setup_wandb_config
)

# Use lowmem datasets instead of regular datasets
from utils.lowmem_prng_data import create_curriculum_lowmem_datasets
from utils.gpt2 import GPT, GPTConfig, GPT_oth_abacus, GPTConfig_abacus, GPT_RoPE
from utils.eval import get_predictions
from utils.file_utils import get_unique_filename, create_base_path, save_results, save_checkpoint
from utils.training_utils import setup, setup_random_seeds, load_excluded_ac_values
from utils.curriculum_utils import (
    parse_moduli_list, get_multi_alpha, create_curriculum_config_custom,
    create_multi_weighted_sampler, evaluate_multi_curriculum_model, save_multi_curriculum_results, 
    print_multi_curriculum_info, setup_phase_based_schedulers, get_multi_curriculum_base_filename,
    save_multi_curriculum_checkpoint
)

import torch._dynamo
import traceback


def load_config_from_yaml(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    print(f"Loaded configuration from: {config_path}")
    print(f"Experiment: {config_dict.get('experiment_name', 'Unnamed')}")
    print(f"Description: {config_dict.get('description', 'No description')}")
    
    return config_dict


def yaml_to_args(config_dict):
    """Convert YAML configuration to argparse-like namespace"""
    config = argparse.Namespace()
    
    # Model configuration
    model_config = config_dict.get('model', {})
    config.n_layer = model_config.get('n_layer', 2)
    config.n_head = model_config.get('n_head', 2)
    config.n_embd = model_config.get('n_embd', 128)
    config.no_rope = model_config.get('no_rope', False)
    
    # Data configuration
    data_config = config_dict.get('data', {})
    config.type = data_config.get('type', 'LCG')
    config.control_bits = str(data_config.get('control_bits', '0'))
    # Normalize: LCG and TLCG do not use control bits
    if config.type in ['LCG', 'TLCG']:
        config.control_bits = '0'
    config.vocab_size = data_config.get('vocab_size', None)
    config.seq_len = data_config.get('seq_len', 513)
    config.n_a = data_config.get('n_a', 32)
    config.n_c = data_config.get('n_c', 32)
    config.n_test_a = data_config.get('n_test_a', 16)
    config.n_test_c = data_config.get('n_test_c', 16)
    config.n_example = data_config.get('n_example', 1)
    config.base = data_config.get('base', None)
    config.digits = data_config.get('digits', 1)
    
    # Curriculum configuration (always custom when loading from YAML)
    curriculum_config = config_dict.get('curriculum', {})
    curriculum_phases = curriculum_config.get('phases', [])
    config.custom_phases = curriculum_phases
    
    # Get transition type from curriculum config
    config.transition_type = curriculum_config.get('transition_type', 'linear')
    
    # Get moduli from curriculum config (new structure)
    config.moduli = curriculum_config.get('moduli', None)
    
    # Get bits_to_keep for each modulus (optional, for PCG types)
    config.moduli_bits_to_keep = curriculum_config.get('bits_to_keep', None)
    
    # Calculate total steps from phases for defaulting
    total_phase_steps = 0
    for phase in curriculum_phases:
        phase_steps = phase.get('phase_steps', 0)
        if phase_steps > 0:
            total_phase_steps += phase_steps
    
    # Training configuration
    training_config = config_dict.get('training', {})
    # Derive total steps strictly from phases
    if total_phase_steps <= 0:
        raise ValueError("No curriculum phases with positive phase_steps found; please specify phases.")
    config.num_steps = total_phase_steps
    
    # Validate curriculum configuration
    if curriculum_phases:
        print(f"Curriculum phases: {len(curriculum_phases)} phases, {total_phase_steps} total steps")
        for i, phase in enumerate(curriculum_phases):
            phase_name = phase.get('name', f'Phase {i+1}')
            phase_steps = phase.get('phase_steps', 0)
            transition_steps = phase.get('transition_steps', 0)
            warmup_steps = phase.get('warmup_steps', 0)
            start_weights = phase.get('start_weights', [])
            end_weights = phase.get('end_weights', [])
            transition_type = phase.get('transition', 'linear')
            # Normalize and validate lr_decay per phase (constant or cosine)
            lr_decay_raw = phase.get('lr_decay', 'constant')
            lr_decay = str(lr_decay_raw).lower()
            if lr_decay not in ('constant', 'cosine'):
                print(f"  Warning: Unknown lr_decay='{lr_decay_raw}' in {phase_name}. Using 'constant'.")
                lr_decay = 'constant'
            phase['lr_decay'] = lr_decay  # persist normalized value
            warmup_info = f" (warmup: {warmup_steps})" if warmup_steps > 0 else ""
            print(f"  {phase_name}: {phase_steps} steps ({transition_steps} transition){warmup_info}, {transition_type}, lr_decay={lr_decay}")
            print(f"    weights: {start_weights} -> {end_weights}")
    else:
        print("Warning: No curriculum phases defined!")
    # Keep transition_type from curriculum config; do not overwrite from training config
    # Only read sampler_update_interval from curriculum section
    config.sampler_update_interval = curriculum_config.get('sampler_update_interval', None)
    
    # Optimizer settings
    config.lr_trgt = float(training_config.get('lr_trgt', 1e-4))
    config.lr_min = float(training_config.get('lr_min', 1e-7))
    # Warmup steps should be specified within each phase
    config.batch_size = training_config.get('batch_size', 64)
    config.grad_acc_steps = training_config.get('grad_acc_steps', 1)
    config.weight_decay = training_config.get('weight_decay', 0.1)
    config.beta1 = training_config.get('beta1', 0.9)
    config.beta2 = training_config.get('beta2', 0.999)
    
    # Evaluation and saving
    config.eval_interval = training_config.get('eval_interval', 100)
    config.save_checkpoints = training_config.get('save_checkpoints', False)
    config.checkpoint_interval = training_config.get('checkpoint_interval', 1000)
    
    # Output configuration
    output_config = config_dict.get('output', {})
    config.save_correctness = output_config.get('save_correctness', False)
    # Default matches parser action='store_true' (False by default)
    config.save_params = output_config.get('save_params', False)
    
    # Pretrained model configuration
    pretrained_config = config_dict.get('pretrained', {})
    config.pretrained_path = pretrained_config.get('pretrained_path', None)
    config.pretrain_m = pretrained_config.get('pretrain_m', None)
    
    # Seeds
    seeds_config = config_dict.get('seeds', {})
    config.main_seed = seeds_config.get('main_seed', 1)
    config.data_seed = seeds_config.get('data_seed', 1)
    
    # Output
    config.results_dir = output_config.get('results_dir', 'results/multi_curriculum')
    
    # Memory efficiency settings
    memory_config = config_dict.get('memory', {})
    config.cache_size = memory_config.get('cache_size', 1000)
    
    # W&B
    setup_wandb_config(config_dict, config)
    
    # Exclude AC paths configuration
    config.exclude_ac_paths = config_dict.get('exclude_ac_paths', None)
    
    # Store original config dict for reference
    config._yaml_config = config_dict
    
    return config


@torch.no_grad()
def load_pretrained_with_vocab_expand(model, checkpoint_path, target_vocab_size, device):
    """Load a pretrained model, expanding vocab if needed.

    - Copies over all matching weights.
    - Special handling for token embeddings and lm_head to support vocab growth.
    - Keeps weight tying between wte and lm_head.
    """
    print(f"Loading pretrained weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("Unsupported checkpoint format: expected dict or dict with 'model_state_dict'")

    embed_key = 'transformer.wte.weight'
    head_key = 'lm_head.weight'

    if embed_key not in state_dict or head_key not in state_dict:
        raise KeyError("Checkpoint missing embedding or lm_head weights; cannot load")

    pretrained_embed = state_dict[embed_key]
    old_vocab_size = pretrained_embed.size(0)
    new_vocab_size = int(target_vocab_size)

    # Remove size-mismatched keys so they don't block partial loading
    state_dict.pop(embed_key, None)
    state_dict.pop(head_key, None)

    # Load remaining weights (non-embedding) with strict=False
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"Warning: Unexpected keys during load: {unexpected}")
    if missing:
        # Embedding/head keys will be missing by design; others may be due to arch diffs
        print(f"Info: Missing keys not loaded (likely due to arch differences): {missing[:10]}{' ...' if len(missing) > 10 else ''}")

    # Now handle embeddings/head with possible expansion
    with torch.no_grad():
        wte_param = model.transformer.wte.weight
        if new_vocab_size < old_vocab_size:
            print(f"Warning: target vocab ({new_vocab_size}) < pretrained vocab ({old_vocab_size}); truncating pretrained weights")
            copy_rows = new_vocab_size
        else:
            copy_rows = old_vocab_size
        # Copy into the front rows; remaining rows keep random init
        wte_param.data[:copy_rows].copy_(pretrained_embed[:copy_rows])
        # Weight tying ensures lm_head shares the same parameter; if not, enforce tie
        if model.lm_head.weight.data.data_ptr() != model.transformer.wte.weight.data.data_ptr():
            model.lm_head.weight = model.transformer.wte.weight

    print(f"Loaded pretrained vocab {old_vocab_size}; target vocab {new_vocab_size}. Copied {copy_rows} rows.")
    return old_vocab_size, new_vocab_size


def train_multi_curriculum(model, optimizer, scheduler, train_datasets, test_loaders, 
                          config, eval_results, device, device_type, 
                          rank=None, world_size=None, master_process=True,
                          train_a_values=None, train_c_values=None):
    """Train the model with multi-modulus curriculum learning using lowmem datasets"""
    try:
        if rank is not None:
            dist.barrier()  # Initial sync point
        
        if master_process:
            print_multi_curriculum_info(config)
            print("Using lowmem datasets with no caching")
        
        # Combine all datasets for training
        train_dataset = ConcatDataset(train_datasets)
        moduli = config.curriculum_config['moduli']
        
        model.train()
        step = 0
        grad_updates = 0
        train_loss = 0
        train_acc = 0
        train_last_acc = 0
        grad_norm_sum = 0
        last_train_correct_vector = None
        
        # Weighted sampler approach (only option supported)
        sampler_update_interval = getattr(config, 'sampler_update_interval', config.eval_interval)
        alpha_weights = get_multi_alpha(0, config.curriculum_config)
        if len(alpha_weights) != len(train_datasets):
            raise ValueError(
                f"Alpha weights length ({len(alpha_weights)}) does not match number of datasets ({len(train_datasets)}). "
                f"Moduli: {config.curriculum_config.get('moduli')}, alpha_weights: {alpha_weights}"
            )
        # Sampler epoch counter for deterministic reseeding on refresh
        sampler_epoch = 0
        sampler = create_multi_weighted_sampler(
            train_datasets, alpha_weights,
            rank, world_size,
            seed=config.main_seed, epoch=sampler_epoch
        )
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                 sampler=sampler, pin_memory=True, drop_last=True,
                                 num_workers=4, prefetch_factor=2)
        train_iter = iter(train_loader)
        
        if master_process:
            print("Using weighted sampler curriculum method with lowmem datasets")
        
        while grad_updates < config.num_steps:
            # Weighted sampler method (only option supported)
            if grad_updates % sampler_update_interval == 0:
                alpha_weights = get_multi_alpha(grad_updates, config.curriculum_config)
                if len(alpha_weights) != len(train_datasets):
                    raise ValueError(
                        f"Alpha weights length ({len(alpha_weights)}) does not match number of datasets ({len(train_datasets)}) at step {grad_updates}. "
                        f"Moduli: {config.curriculum_config.get('moduli')}, alpha_weights: {alpha_weights}"
                    )
                sampler_epoch += 1
                if master_process:
                    print(f"Refreshing sampler at step {grad_updates} (epoch {sampler_epoch}). Weights: {alpha_weights}")
                    
                    # Clear caches periodically to manage memory
                    if sampler_epoch % 10 == 0:
                        for ds in train_datasets:
                            if hasattr(ds, 'clear_cache'):
                                ds.clear_cache()
                        print(f"Cleared dataset caches at epoch {sampler_epoch}")
                
                sampler = create_multi_weighted_sampler(
                    train_datasets, alpha_weights,
                    rank, world_size,
                    seed=config.main_seed, epoch=sampler_epoch
                )
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                         sampler=sampler, pin_memory=True, drop_last=True,
                                         num_workers=4, prefetch_factor=2)
                train_iter = iter(train_loader)
            
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
                loss = loss / config.grad_acc_steps
            
            with torch.no_grad():
                train_loss += loss.item() * config.grad_acc_steps
                correct = (logits.argmax(dim=-1) == y)
                train_acc += correct.sum().item()
                train_last_acc += torch.sum(torch.all(correct[:,-config.digits:], dim=1)).item()
                # If this micro-batch will complete the final grad update, capture per-number correctness
                will_update = ((step + 1) % config.grad_acc_steps == 0)
                is_last_update = will_update and (grad_updates + 1 == config.num_steps)
                if is_last_update:
                    digits = int(getattr(config, 'digits', 1))
                    token_correct = correct[:, digits-1:]
                    number_correct = token_correct[:, ::digits].clone()
                    for d in range(1, digits):
                        number_correct &= token_correct[:, d::digits]
                    last_train_correct_vector = number_correct.float().mean(dim=0).cpu().numpy()
                
                del logits, correct
                torch.cuda.empty_cache()
            
            loss.backward()
            step += 1
            
            if step % config.grad_acc_steps == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_norm_sum += norm.item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                grad_updates += 1
                
                # Checkpoint saving
                if (master_process and getattr(config, 'save_checkpoints', False) and 
                    hasattr(config, 'checkpoint_interval') and
                    grad_updates % config.checkpoint_interval == 0):
                    save_multi_curriculum_checkpoint(model.module if isinstance(model, DDP) else model, 
                                                   optimizer, scheduler, grad_updates, eval_results, config, 
                                                   world_size if rank is not None else 1, rank is not None)
                
                # Regular evaluation
                if grad_updates % config.eval_interval == 0:
                    if rank is not None:
                        dist.barrier()
                    
                    model.eval()
                    # Provide the current batch to compute lightweight train correctness at the end
                    # Use captured per-number correctness vector at final step (if available)
                    train_correct_vector = last_train_correct_vector if grad_updates >= config.num_steps else None
                    test_metrics = evaluate_multi_curriculum_model(model, test_loaders, 
                                                               device, device_type, config,
                                                               save_correctness=False,
                                                               rank=rank if rank is not None else None,
                                                               world_size=world_size if rank is not None else None,
                                                               master_process=master_process,
                                                               train_datasets=train_datasets,
                                                               train_batch=None,
                                                               train_correct_vector=train_correct_vector)
                    
                    train_loss_avg = train_loss / config.eval_interval
                    train_acc_avg = train_acc / (config.eval_interval * config.grad_acc_steps * config.batch_size * config.context_len)
                    train_last_acc_avg = train_last_acc / (config.eval_interval * config.grad_acc_steps * config.batch_size)
                    avg_grad_norm = grad_norm_sum / config.eval_interval
                    
                    if rank is not None:
                        # Extract scalar metrics for distributed reduction (single reduction here)
                        metrics_list = [train_loss_avg, train_acc_avg, train_last_acc_avg, avg_grad_norm]
                        for m in moduli:
                            metrics_list.extend([test_metrics[f'm{m}']['loss'], test_metrics[f'm{m}']['acc'], test_metrics[f'm{m}']['last_acc']])
                        metrics = torch.tensor(metrics_list, device=device)
                        dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
                        train_loss_avg, train_acc_avg, train_last_acc_avg, avg_grad_norm = metrics[:4].tolist()
                        # Reconstruct test metrics
                        test_metrics = {}
                        idx = 4
                        for m in moduli:
                            test_metrics[f'm{m}'] = {
                                'loss': metrics[idx].item(),
                                'acc': metrics[idx+1].item(),
                                'last_acc': metrics[idx+2].item()
                            }
                            idx += 3
                        dist.barrier()
                    
                    if master_process:
                        current_lr = scheduler.get_last_lr()[0]
                        alpha_weights = get_multi_alpha(grad_updates, config.curriculum_config)
                        
                        # Prepare eval results row
                        eval_row = [grad_updates, current_lr, train_loss_avg, train_acc_avg, train_last_acc_avg, avg_grad_norm]
                        
                        # Add alpha weights
                        eval_row.extend(alpha_weights)
                        
                        # Add test metrics
                        for m in moduli:
                            eval_row.extend([test_metrics[f'm{m}']['loss'], test_metrics[f'm{m}']['acc'], test_metrics[f'm{m}']['last_acc']])
                        
                        eval_results.append(eval_row)
                        
                        # Print training progress
                        weights_str = ', '.join([f'm{m}={w:.3f}' for m, w in zip(moduli, alpha_weights)])
                        print(f"Step {grad_updates:4d} | lr: {current_lr:.2e} | train loss: {train_loss_avg:.6f} | train acc: {train_acc_avg:.4f} | train last acc: {train_last_acc_avg:.4f} | grad norm: {avg_grad_norm:.4f}")
                        print(f"Weights: [{weights_str}]")
                        for m in moduli:
                            print(f"  m={m:>5} TEST | loss: {test_metrics[f'm{m}']['loss']:.6f} | acc: {test_metrics[f'm{m}']['acc']:.4f} | last acc: {test_metrics[f'm{m}']['last_acc']:.4f}")
                        
                        # Log to W&B if available
                        log_training_metrics(config, grad_updates, train_loss_avg, train_acc_avg,
                                           train_last_acc_avg, avg_grad_norm, current_lr,
                                           alpha_weights, moduli, test_metrics)
                    
                    # Reset counters
                    train_loss = 0
                    train_acc = 0
                    train_last_acc = 0
                    grad_norm_sum = 0
                    model.train()
        
        # Final evaluation with correctness saving
        if config.save_correctness:
            if master_process:
                print("Performing final evaluation with correctness saving...")
            
            try:
                if rank is not None:
                    dist.barrier()
                
                model.eval()
                # At final evaluation, recompute train correctness from the last observed training batch if available
                test_metrics = evaluate_multi_curriculum_model(model, test_loaders, 
                                                           device, device_type, config,
                                                           save_correctness=True,
                                                           rank=rank if rank is not None else None,
                                                           world_size=world_size if rank is not None else None,
                                                           master_process=master_process,
                                                           train_datasets=train_datasets,
                                                           train_batch=None,
                                                           train_correct_vector=last_train_correct_vector)
                model.train()
                
                if rank is not None:
                    dist.barrier()
                    
            except Exception as e:
                if master_process:
                    print(f"Warning: Final evaluation failed: {str(e)}")
                    print("Continuing with training completion...")
                if rank is not None:
                    dist.barrier()
        
        # Final cache cleanup
        if master_process:
            print("Clearing dataset caches...")
            for ds in train_datasets:
                if hasattr(ds, 'clear_cache'):
                    ds.clear_cache()
        
        return eval_results
    
    except Exception as e:
        print(f"Error in training: {str(e)}")
        print(traceback.format_exc())
        raise e


def main():
    parser = argparse.ArgumentParser(description='Memory-efficient (lowmem) multi-modulus curriculum training for PRNG tasks')
    
    # Configuration file option
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file')
    
    # Memory efficiency options (no longer needed - no caching)
    # parser.add_argument('--cache_size', type=int, default=1000,
    #                     help='Number of sequences to cache per dataset (default: 1000)')
    
    # Model architecture
    parser.add_argument('--type', type=str, default='LCG',
                        help='PRNG type (LCG, etc.)')
    parser.add_argument('--moduli', type=str, default='256,1024,4096',
                        help='Comma-separated list of moduli in increasing difficulty order')
    parser.add_argument('--control_bits', type=str, default='0',
                        help='Number of control bits')
    parser.add_argument('--vocab_size', type=int, default=None,
                        help='Vocabulary size')
    parser.add_argument('--seq_len', type=int, default=513,
                        help='Sequence length')
    parser.add_argument('--n_a', type=int, default=32,
                        help='Number of training a values')
    parser.add_argument('--n_c', type=int, default=32,
                        help='Number of training c values')
    parser.add_argument('--n_test_a', type=int, default=16,
                        help='Number of test a values')
    parser.add_argument('--n_test_c', type=int, default=16,
                        help='Number of test c values')
    parser.add_argument('--n_example', type=int, default=1,
                        help='Number of examples per (a,c) pair')
    parser.add_argument('--base', type=int, default=None,
                        help='Base for number representation')
    parser.add_argument('--digits', type=int, default=1,
                        help='Number of digits to check for accuracy')
    
    # Model parameters
    parser.add_argument('--n_layer', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=128,
                        help='Embedding dimension')
    
    # Curriculum parameters
    # curriculum_type is always custom
    parser.add_argument('--transition_type', type=str, default='linear',
                        choices=['linear', 'cosine', 'step', 'exp'],
                        help='Type of transition between curriculum phases')
    # Only custom curriculum is supported
    # Removed smooth option - only weighted sampling supported
    parser.add_argument('--sampler_update_interval', type=int, default=None,
                        help='How often to update the sampler (in steps)')
    
    # Custom curriculum (if curriculum_type='custom')
    parser.add_argument('--custom_phases', type=str, default=None,
                        help='Custom phase configuration as JSON string (required if not using --config)')
    
    # Training parameters
    parser.add_argument('--main_seed', type=int, default=1,
                        help='Random seed for model initialization and training')
    parser.add_argument('--data_seed', type=int, default=1,
                        help='Random seed for data generation')
    # num_steps is derived from curriculum phases and not set via CLI
    parser.add_argument('--lr_trgt', type=float, default=1e-4,
                        help='Target (max) learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-7,
                        help='Minimum learning rate for cosine annealing')
    # Warmup steps should be specified within each phase
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--grad_acc_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Evaluation interval')
    
    # Other parameters
    parser.add_argument('--results_dir', type=str, default='results/multi_curriculum_lazy',
                        help='Output directory for results')
    parser.add_argument('--save_checkpoints', action='store_true',
                        help='Enable checkpoint saving')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Checkpoint saving interval')
    parser.add_argument('--save_params', action='store_true',
                        help='Save final model parameters')
    parser.add_argument('--no_rope', action='store_true',
                        help='Disable RoPE positional embedding')
    
    # Pretrained loading
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained checkpoint/model to initialize from')
    
    # Exclude AC values configuration
    parser.add_argument('--exclude_ac_paths', type=str, nargs='*', default=None,
                        help='List of paths to AC files to exclude from test set generation')
    
    # W&B logging
    # Add W&B arguments
    add_wandb_args(parser)
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config is not None:
        # Load from YAML file
        config_dict = load_config_from_yaml(args.config)
        config = yaml_to_args(config_dict)
        print(f"Using configuration from: {args.config}")
        
        # Cache size no longer used - removed for pure lowmem approach
        # if args.cache_size != 1000:  # Check if different from default
        #     config.cache_size = args.cache_size
        #     print(f"Overriding cache_size from config with command-line value: {args.cache_size}")
        
        # Override pretrained_path from command line if provided
        if args.pretrained_path is not None:
            config.pretrained_path = args.pretrained_path
            print(f"Overriding pretrained_path from config with command-line value: {args.pretrained_path}")
        
        # Override exclude_ac_paths from command line if provided
        if args.exclude_ac_paths is not None:
            config.exclude_ac_paths = args.exclude_ac_paths
            print(f"Overriding exclude_ac_paths from config with command-line value: {args.exclude_ac_paths}")
        
        # Ensure args has pretrained_path for later access
        args.pretrained_path = getattr(config, 'pretrained_path', None)
    else:
        # Use command line arguments
        config = args
        moduli = parse_moduli_list(config.moduli)
        config.moduli = moduli
        print(f"Using command-line configuration with moduli: {moduli}")
        if config.custom_phases is None:
            raise ValueError("--custom_phases must be provided when not using --config")
        import json
        phases_config = json.loads(config.custom_phases)
        curriculum_config = create_curriculum_config_custom(
            moduli=moduli,
            phases_config=phases_config,
            transition_type=config.transition_type
        )
        # Ensure downstream code has consistent attributes
        config.custom_phases = phases_config
        # Derive total steps strictly from phases
        phase_total = sum(int(p.get('phase_steps', 0)) for p in phases_config)
        if phase_total <= 0:
            raise ValueError("custom_phases must include at least one phase with positive phase_steps")
        config.num_steps = phase_total
    
    # For YAML configs, create curriculum configuration
    if args.config is not None:
        curriculum_config = create_curriculum_config_custom(
            moduli=config.moduli,
            phases_config=config.custom_phases,
            transition_type=config.transition_type
        )
    
    config.curriculum_config = curriculum_config
    
    # Set sampler update interval
    if config.sampler_update_interval is None:
        config.sampler_update_interval = config.eval_interval
    
    # Set base and vocab_size
    if config.base is None:
        config.base = max(config.moduli)
    if config.vocab_size is None:
        config.vocab_size = config.base
    
    # Calculate context length
    config.context_len = (config.seq_len * config.digits) - 1
    
    # Set up distributed training
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    
    
    ddp = int(os.environ["WORLD_SIZE"]) != 1
    
    if ddp:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ.get("SLURM_PROCID", "0"))
        gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        
        torch.cuda.set_device(local_rank)
        
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        
        master_process = rank == 0
        device = f"cuda:{local_rank}"
        device_type = "cuda"
        
        print(f"DDP SETUP: host: {gethostname()}, rank: {rank}/{world_size - 1}, local_rank: {local_rank}")
        if master_process:
            print(f"DDP initialized: {dist.is_initialized()}", flush=True)
    else:
        ddp = False
        rank = 0
        local_rank = 0
        world_size = 1
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_type = "cuda" if device.startswith("cuda") else "cpu"
        master_process = True
        print(f"SINGLE PROCESS MODE: host: {gethostname()}")
    # Expose world_size on config for downstream filename generation and logging
    config.world_size = world_size
    
    # Set up environment
    setup_random_seeds(config.main_seed)
    torch.set_float32_matmul_precision("high")
    
    # Create output directory
    if master_process:
        os.makedirs(config.results_dir, exist_ok=True)
        
    # Initialize W&B
    init_wandb(config, master_process)
    
    # Load excluded AC values if specified
    excluded_a, excluded_c = load_excluded_ac_values(config.exclude_ac_paths, master_process)
    
    # Generate lowmem datasets for all moduli
    if master_process:
        print("="*80)
        print("GENERATING LOWMEM MULTI-MODULUS CURRICULUM DATA")
        for i, m in enumerate(config.moduli):
            # Use custom bits_to_keep if provided, otherwise default to log2(m)
            if hasattr(config, 'moduli_bits_to_keep') and config.moduli_bits_to_keep is not None:
                if i < len(config.moduli_bits_to_keep):
                    bits_to_keep = config.moduli_bits_to_keep[i]
                else:
                    bits_to_keep = int(math.ceil(math.log2(m)))
                    print(f"  Warning: No bits_to_keep specified for modulus {i+1}, using default: {bits_to_keep}")
            else:
                bits_to_keep = int(math.ceil(math.log2(m)))
            print(f"  Modulus {i+1}: {config.type} with m={m}, bits_to_keep={bits_to_keep}")
        print("="*80)
    
    t0 = time.time()
    
    # Create datasets
    train_datasets, test_loaders, train_a_values, train_c_values = create_curriculum_lowmem_datasets(
        config, master_process, ddp, rank if ddp else None, world_size if ddp else None,
        num_workers=getattr(config, 'num_workers', 4),  # Default to 4 workers for server performance
        excluded_a=excluded_a, excluded_c=excluded_c
    )
    
    t1 = time.time()
    
    if master_process:
        print(f"Data generation completed in {t1-t0:.2f} seconds")
        for i, (m, dataset) in enumerate(zip(config.moduli, train_datasets)):
            print(f"  m={m}: {len(dataset)} train sequences")
    
    # Initialize model
    if not config.no_rope:
        model = GPT_RoPE(GPTConfig(
            block_size=config.context_len,
            n_embd=config.n_embd,
            n_head=config.n_head,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer
        ))
    else:
        if int(getattr(config, 'digits', 1)) > 1:
            model = GPT_oth_abacus(GPTConfig_abacus(
                block_size=config.context_len,
                n_embd=config.n_embd,
                n_head=config.n_head,
                vocab_size=config.vocab_size,
                n_layer=config.n_layer,
                digits=int(getattr(config, 'digits', 1))
            ))
        else:
            model = GPT(GPTConfig(
                block_size=config.context_len,
                n_embd=config.n_embd,
                n_head=config.n_head,
                vocab_size=config.vocab_size,
                n_layer=config.n_layer
            ))
    
    model.to(device)
    # Report model variant, structure, and parameter counts (master only)
    if master_process:
        variant = (
            'GPT_RoPE' if not config.no_rope else (
                'GPT_oth_abacus' if int(getattr(config, 'digits', 1)) > 1 else 'GPT'
            )
        )
        print("="*80)
        print(f"MODEL: {variant} | type={config.type} | digits={int(getattr(config, 'digits', 1))}")
        print("-"*80)
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")
        print("="*80)
        print("LOWMEM FEATURES:")
        print(f"  - Using on-demand sequence generation")
        print(f"  - No caching - pure memory efficiency")
        print(f"  - Sequences generated on-demand")
        print("="*80)
    
    # Wrap model (skip compile for Python 3.12+)
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    raw_model = model.module if ddp else model
    
    # Set up optimizer
    # Suppress optimizer creation prints on non-master ranks to avoid duplicates
    if ddp and not master_process:
        import contextlib
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            optimizer = raw_model.configure_optimizers(
                weight_decay=config.weight_decay,
                learning_rate=config.lr_trgt,
                beta1=config.beta1,
                beta2=config.beta2,
                device=device
            )
    else:
        optimizer = raw_model.configure_optimizers(
            weight_decay=config.weight_decay,
            learning_rate=config.lr_trgt,
            beta1=config.beta1,
            beta2=config.beta2,
            device=device
        )
    
    # Set up scheduler with phase-based warmup
    scheduler = setup_phase_based_schedulers(optimizer, config)
    
    # Optional: load pretrained and expand vocab if needed
    if getattr(args, 'pretrained_path', None):
        try:
            if master_process:
                print("="*80)
                print("LOADING PRETRAINED MODEL")
                print("="*80)
            old_vocab, new_vocab = load_pretrained_with_vocab_expand(raw_model, args.pretrained_path, config.vocab_size, device)
            if master_process:
                print(f"Pretrained model loaded successfully:")
                print(f"  - Original vocab size: {old_vocab}")
                print(f"  - Target vocab size: {new_vocab}")
                print(f"  - Vocab expansion: {'Yes' if new_vocab > old_vocab else 'No'}")
                print("="*80)
        except Exception as e:
            if master_process:
                print(f"Error loading pretrained weights: {e}")
                print("Continuing with random initialization...")
                import traceback
                traceback.print_exc()
    
    if master_process:
        print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        if getattr(args, 'pretrained_path', None):
            print("Starting lowmem multi-modulus curriculum fine-tuning...")
        else:
            print("Starting multi-modulus curriculum training...")
    
    # Train the model
    eval_results = []
    train_multi_curriculum(model, optimizer, scheduler, train_datasets, test_loaders,
                          config, eval_results, device, device_type,
                          rank if ddp else None, world_size if ddp else None, master_process,
                          train_a_values, train_c_values)
    
    # Save results
    if master_process:
        print("Training completed. Saving results...")
        save_multi_curriculum_results(config, eval_results, config.results_dir)
        
        if config.save_params:
            # Save final model using the same base filename as evaluation results
            base_filename = get_multi_curriculum_base_filename(config)
            final_model_path = os.path.join(config.results_dir, f"multi_curriculum_model_lazy_{base_filename}.pt")
            final_model_path = get_unique_filename(final_model_path)
            
            torch.save({
                'model_state_dict': raw_model.state_dict(),
                'config': vars(config),
                'final_step': config.num_steps,
                'lowmem': True
            }, final_model_path)
            print(f"Saved final model to {final_model_path}")
            
            # Save training a,c values separately with data-only filename
            from utils.curriculum_utils import get_training_ac_values_filename
            ac_filename = get_training_ac_values_filename(config)
            ac_path = os.path.join(config.results_dir, f"{ac_filename}.pt")

            
            # Create a dictionary mapping moduli to their training a and c values
            training_ac_values = {}
            for i, m in enumerate(config.moduli):
                training_ac_values[m] = {
                    'train_a': train_a_values[i],
                    'train_c': train_c_values[i]
                }
            
            torch.save({
                'training_ac_values': training_ac_values,
                'config': {
                    'type': config.type,
                    'moduli': config.moduli,
                    'n_a': config.n_a,
                    'n_c': config.n_c,
                    'n_example': config.n_example,
                    'vocab_size': config.vocab_size,
                    'seq_len': config.seq_len,
                    'control_bits': getattr(config, 'control_bits', '0'),
                    'data_seed': config.data_seed,
                    'main_seed': config.main_seed
                }
            }, ac_path)
            print(f"Saved training a,c values to {ac_path}")
        
        finish_wandb(config)
    
    # Clean up
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
