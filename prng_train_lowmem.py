import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import time
import math
import numpy as np
import pandas as pd
import platform
import psutil

# Try to import wandb, but optional
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception as e:
    print(f"Warning: wandb not available: {e}")
    WANDB_AVAILABLE = False

# Use lowmem data generation
from utils.lowmem_prng_data import generate_lowmem_data, validate_prng_parameters
from utils.gpt2 import GPT, GPTConfig, GPT_oth_abacus, GPTConfig_abacus, GPT_RoPE
from utils.eval import get_predictions
from utils.file_utils import get_unique_filename, create_base_path, save_results, save_checkpoint
from utils.training_utils import setup, setup_random_seeds, evaluate_model, load_excluded_ac_values
from utils.wandb_utils import init_wandb, finish_wandb, add_wandb_args

import argparse
import sys
import os
from socket import gethostname


import torch._dynamo


def train(model, optimizer, scheduler, train_loader, test_loader, num_epoch, eval_results, device, config, master_process, device_type, ddp, rank, world_size, raw_model, t0, t1, effective_batch_size, train_a, train_c, test_a, test_c, train_dataset, test_dataset, per_test_loaders=None, per_type_labels=None, per_type_eval_results=None):
    model.train()
    step = 0
    grad_updates = 0
    train_loss = 0
    train_acc = 0
    train_last_acc = 0
    grad_norm_sum = 0
    
    # Evaluate model before training
    if master_process:
        print("Evaluating model before training...")
    
    pretrain_test_loss = 0.0
    pretrain_test_acc = 0
    pretrain_test_last_acc = 0
    
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Guard autocast by device type
            if device_type == 'cuda':
                autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            elif device_type == 'cpu':
                autocast_ctx = torch.autocast(device_type='cpu', dtype=torch.float16)
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()
            with autocast_ctx:
                logits, loss = model(x, y) 
            pretrain_test_loss += loss.item()
            preds = logits.argmax(dim=-1)
            pretrain_test_acc += torch.sum(preds == y).item()
            
            # Check last `digits` tokens (equivalent to last token when digits=1)
            pretrain_test_last_acc += torch.sum(torch.all(preds[:,-config.digits:]==y[:,-config.digits:], dim=1)).item()
            
            del logits, loss, preds
    
    # Avoid division by zero
    if len(test_loader) == 0:
        pretrain_test_loss = 0.0
        pretrain_test_acc = 0.0
        pretrain_test_last_acc = 0.0
    else:
        pretrain_test_loss = pretrain_test_loss / len(test_loader) 
        pretrain_test_acc = pretrain_test_acc / len(test_loader) / config.batch_size / config.context_len
        pretrain_test_last_acc = pretrain_test_last_acc / len(test_loader) / config.batch_size
    
    if ddp:
        metrics = torch.tensor([pretrain_test_loss, pretrain_test_acc, pretrain_test_last_acc], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
        pretrain_test_loss, pretrain_test_acc, pretrain_test_last_acc = metrics.tolist()
    
    if master_process:
        print(f"INITIAL EVALUATION | test loss: {pretrain_test_loss:.6f} | test acc: {pretrain_test_acc:.4f} | test last acc: {pretrain_test_last_acc:.4f}")
        eval_results.append([0, float('nan'), pretrain_test_loss, float('nan'), pretrain_test_acc, float('nan'), pretrain_test_last_acc, float('nan')])
        
        # Log initial evaluation to W&B
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "learning_curves/test_loss": pretrain_test_loss,
                "learning_curves/test_acc": pretrain_test_acc,
                "learning_curves/test_last_acc": pretrain_test_last_acc,
            }, step=0, commit=True)
    
    # Per-type initial evaluation (if configured)
    if per_test_loaders:
        per_type_metrics_local = {}
        for label in per_type_labels:
            loader = per_test_loaders[label]
            ploss, pacc, plast = evaluate_model(model, loader, device, device_type, config)
            per_type_metrics_local[label] = (ploss, pacc, plast)
        if master_process:
            print("Per-type INITIAL EVALUATION:")
            for label in per_type_labels:
                ploss, pacc, plast = per_type_metrics_local[label]
                print(f"  [{label}] loss: {ploss:.6f} | acc: {pacc:.4f} | last acc: {plast:.4f}")
            if config.use_wandb and WANDB_AVAILABLE:
                log_dict = {}
                for label in per_type_labels:
                    ploss, pacc, plast = per_type_metrics_local[label]
                    log_dict[f"per_type/{label}/loss"] = ploss
                    log_dict[f"per_type/{label}/acc"] = pacc
                    log_dict[f"per_type/{label}/last_acc"] = plast
                wandb.log(log_dict, step=0, commit=True)
            if per_type_labels:
                row = [0]
                for label in per_type_labels:
                    m = per_type_metrics_local.get(label, (float('nan'), float('nan'), float('nan')))
                    row.extend(list(m))
                per_type_eval_results.append(row)

                # Checkpoint saving
    if (master_process and config.save_checkpoints):
        save_checkpoint(raw_model, optimizer, scheduler, grad_updates, eval_results, config, world_size, ddp)

    # Training loop
    model.train()
    
    for epoch in range(num_epoch):
        if ddp:
            train_loader.sampler.set_epoch(epoch)
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # Mark step begin for CUDA graphs if available (avoid under DDP to reduce sync issues)
            if device_type == 'cuda' and (not ddp) and hasattr(torch, 'compiler') and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                torch.compiler.cudagraph_mark_step_begin()
            
            # Guard autocast by device type
            if device_type == 'cuda':
                autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            elif device_type == 'cpu':
                autocast_ctx = torch.autocast(device_type='cpu', dtype=torch.float16)
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()
            with autocast_ctx:
                logits, loss = model(x, y) 
                loss = loss / config.grad_acc_steps
            
            with torch.no_grad():
                train_loss += loss.item()
                correct = (logits.argmax(dim=-1) == y)
                train_acc += correct.sum().item()
                
                # Check last `digits` tokens (equivalent to last token when digits=1)
                train_last_acc += torch.sum(torch.all(correct[:,-config.digits:], dim=1)).item()
                
                del logits, correct
            
            loss.backward()
            step += 1
            
            # Update weights after accumulating gradients
            if step % config.grad_acc_steps == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_norm_sum += norm.item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                grad_updates += 1
                
                if grad_updates % config.eval_interval == 0:
                    test_loss, test_acc, test_last_acc = evaluate_model(model, test_loader, device, device_type, config)

                    train_loss = train_loss / config.eval_interval / config.grad_acc_steps
                    train_acc = train_acc / (config.eval_interval * config.grad_acc_steps * config.batch_size * config.context_len)
                    train_last_acc = train_last_acc / (config.eval_interval * config.grad_acc_steps * config.batch_size)
                    avg_grad_norm = grad_norm_sum / config.eval_interval
                    
                    if ddp:
                        metrics = torch.tensor([train_loss, train_acc, train_last_acc, test_loss, test_acc, test_last_acc, avg_grad_norm], device=device)
                        dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
                        train_loss, train_acc, train_last_acc, test_loss, test_acc, test_last_acc, avg_grad_norm = metrics.tolist()
                    if master_process:
                        print(f"rank {rank} | step {grad_updates:4d} | batch {epoch} | train loss: {train_loss:.6f} | test loss: {test_loss:.6f} | train acc: {train_acc:.4f} | test acc: {test_acc:.4f} | train last acc: {train_last_acc:.4f} | test last acc: {test_last_acc:.4f} | grad norm: {avg_grad_norm:.4f}")
                        eval_results.append([grad_updates, train_loss, test_loss, train_acc, test_acc, train_last_acc, test_last_acc, avg_grad_norm])
                        
                        # Log metrics to W&B
                        if config.use_wandb and WANDB_AVAILABLE:
                            log_dict = {
                                "training/step": grad_updates,
                                "training/epoch": epoch,
                                "learning_curves/train_loss": train_loss,
                                "learning_curves/test_loss": test_loss,
                                "learning_curves/train_acc": train_acc,
                                "learning_curves/test_acc": test_acc,
                                "learning_curves/train_last_acc": train_last_acc,
                                "learning_curves/test_last_acc": test_last_acc,
                                "learning_curves/grad_norm": avg_grad_norm,
                                "learning_curves/learning_rate": optimizer.param_groups[0]['lr'],
                            }
                            
                            # Add GPU memory info if using CUDA
                            if device.startswith('cuda'):
                                log_dict.update({
                                    "system/gpu_memory_allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
                                    "system/gpu_memory_reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
                                })
                            
                            # Add CPU usage
                            cpu_percent = psutil.cpu_percent()
                            if cpu_percent is not None:
                                log_dict["cpu_percent"] = cpu_percent
                            
                            wandb.log(log_dict, step=grad_updates, commit=True)

                        # Per-type evaluation (if configured)
                        if per_test_loaders:
                            per_type_metrics_local = {}
                            for label in per_type_labels:
                                loader = per_test_loaders[label]
                                ploss, pacc, plast = evaluate_model(model, loader, device, device_type, config)
                                per_type_metrics_local[label] = (ploss, pacc, plast)
                            if master_process:
                                for label in per_type_labels:
                                    ploss, pacc, plast = per_type_metrics_local[label]
                                    print(f"    [{label}] test | loss: {ploss:.6f} | acc: {pacc:.4f} | last acc: {plast:.4f}")
                                if config.use_wandb and WANDB_AVAILABLE:
                                    log_dict = {}
                                    for label in per_type_labels:
                                        ploss, pacc, plast = per_type_metrics_local[label]
                                        log_dict[f"per_type/{label}/loss"] = ploss
                                        log_dict[f"per_type/{label}/acc"] = pacc
                                        log_dict[f"per_type/{label}/last_acc"] = plast
                                    wandb.log(log_dict, step=grad_updates, commit=True)
                                if per_type_labels:
                                    row = [grad_updates]
                                    for label in per_type_labels:
                                        m = per_type_metrics_local.get(label, (float('nan'), float('nan'), float('nan')))
                                        row.extend(list(m))
                                    per_type_eval_results.append(row)
                    # No explicit barriers; rely on both ranks executing same eval path
                    
                    train_loss = 0
                    train_acc = 0
                    train_last_acc = 0
                    grad_norm_sum = 0
                    
                    if grad_updates >= config.num_steps:
                        return
            
            # Checkpoint saving
            if (master_process and config.save_checkpoints and 
                step % config.grad_acc_steps == 0 and
                grad_updates % config.checkpoint_interval == 0):
                save_checkpoint(raw_model, optimizer, scheduler, grad_updates, eval_results, config, world_size, ddp)

def get_parser():
    """Create and return the argument parser for PRNG training"""
    parser = argparse.ArgumentParser(description='Memory-efficient (lowmem) PRNG training with transformer models')
    
    # Random seed configuration
    parser.add_argument('--main_seed', type=int, default=1,
                        help='Random seed for model initialization and training')
    parser.add_argument('--data_seed', type=int, default=1,
                        help='Random seed for data generation')
    
    # Data configuration
    parser.add_argument('--type', type=str, default='LCG',
                        help='PRNG type (LCG, TLCG, RS, RR, XSHRR, XSHRS, XSLRR)')
    parser.add_argument('--m', type=int, default=65536,
                        help='Modulus for the PRNG')
    parser.add_argument('--control_bits', type=str, default='',
                        help='Control bits for PCG variants (comma-separated for multiple)')
    parser.add_argument('--bits_to_keep', type=int, default=None,
                        help='Number of bits to keep in truncated generators')
    parser.add_argument('--vocab_size', type=int, default=None,
                        help='Vocabulary size (auto-determined if None)')
    parser.add_argument('--seq_len', type=int, default=513,
                        help='Sequence length for training')
    parser.add_argument('--base', type=int, default=None,
                        help='Base for number representation')
    parser.add_argument('--digits', type=int, default=1,
                        help='Number of digits to check for accuracy')
    
    # Dataset size configuration
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
    
    # Model architecture
    parser.add_argument('--n_layer', type=int, default=1,
                        help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=1,
                        help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--no_rope', action='store_true',
                        help='Disable RoPE positional embedding')
    
    # Training configuration
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Total number of training steps')
    parser.add_argument('--warm_steps', type=int, default=10,
                        help='Number of warmup steps for learning rate')
    parser.add_argument('--lr_trgt', type=float, default=0.001,
                        help='Target (maximum) learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-7,
                        help='Minimum learning rate for cosine annealing')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--grad_acc_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--weight_decay', type=float, default=0.5,
                        help='Weight decay for regularization')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam optimizer beta1 parameter')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam optimizer beta2 parameter')
    
    # Evaluation configuration
    parser.add_argument('--eval_interval', type=int, default=20,
                        help='Steps between evaluations')
    
    # Output and checkpointing
    parser.add_argument('--results_dir', type=str, default='results/train',
                        help='Directory to save results and checkpoints')
    parser.add_argument('--save_correctness', action='store_true',
                        help='Save prediction correctness analysis')
    parser.add_argument('--save_params', action='store_true',
                        help='Save final model parameters')
    parser.add_argument('--save_test_values', action='store_true',
                        help='Save test a and c values used for evaluation')
    parser.add_argument('--save_checkpoints', action='store_true',
                        help='Enable checkpoint saving during training')
    parser.add_argument('--checkpoint_interval', type=int, default=2000,
                        help='Steps between checkpoint saves')
    
    # Performance configuration
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of DataLoader workers for data loading')
    
    # Exclude AC values configuration
    parser.add_argument('--exclude_ac_paths', type=str, nargs='*', default=None,
                        help='List of paths to AC files to exclude from test set generation')
    
    # W&B logging
    add_wandb_args(parser)
    
    return parser


def setup_config(config):
    """Setup and validate configuration parameters"""
    # Parse multiple types separated by '+' 
    if '+' in config.type:
        config.type_list = [t.strip() for t in config.type.split('+')]
    else:
        config.type_list = [config.type]

    # Set LCG-specific parameters for any LCG type in the list
    if 'LCG' in config.type_list:
        import math
        if config.bits_to_keep is None:
            config.bits_to_keep = int(math.ceil(math.log2(config.m)))
        if config.control_bits == '':
            config.control_bits = '0'
    elif 'TLCG' in config.type_list:
        if config.control_bits == '':
            config.control_bits = '0'

    # Ensure bits_to_keep is set for multi-type configurations
    if config.bits_to_keep is None:
        import math
        bit_length = int(math.ceil(math.log2(config.m)))
        # Use bit_length for all types since TLCG now supports bits_to_keep = bit_length
        config.bits_to_keep = bit_length
        print(f"Warning: bits_to_keep not specified for multi-type configuration, using {config.bits_to_keep}")
        print(f"Set bits_to_keep = {config.bits_to_keep}")

    # Set default base if not provided
    if config.base is None:
        if 'LCG' in config.type_list:
            config.base = config.m
        else:
            config.base = 2 ** config.bits_to_keep

    # Set vocab_size to base if not explicitly provided
    if config.vocab_size is None:
        config.vocab_size = config.base
    elif config.vocab_size < config.base:
        print(f"WARNING: vocab_size < base, setting vocab_size to base: {config.base}")
        config.vocab_size = config.base

    # Calculate context length
    config.context_len = (config.seq_len * config.digits) - 1
    
    return config


def setup_distributed():
    """Setup distributed training environment"""
    # Ensure WORLD_SIZE is set and define ddp early to avoid issues in DataLoader workers
    os.environ.setdefault("WORLD_SIZE", "1")
    ddp = int(os.environ.get("WORLD_SIZE", "1")) != 1  # is this a ddp run?

    if ddp:
        # Set up distributed training
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ.get("SLURM_PROCID", "0"))
        gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        assert gpus_per_node == torch.cuda.device_count() 
        setup(rank, world_size)  
        print(f"DDP SETUP: host: {gethostname()}, rank: {rank}/{world_size-1}, local_rank: {local_rank}")
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        
        master_process = rank == 0
        if master_process: print(f"DDP initialized: {dist.is_initialized()}", flush=True)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"SINGLE PROCESS MODE: Using device: {device}")
        
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    
    return ddp, rank, local_rank, world_size, master_process, device, device_type


def main():
    """Main training function"""
    # Parse arguments
    parser = get_parser()
    config = parser.parse_args()
    
    # Setup configuration
    config = setup_config(config)
    
    # Setup distributed training
    ddp, rank, local_rank, world_size, master_process, device, device_type = setup_distributed()
    
    # Set up environment
    setup_random_seeds(config.main_seed)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(config.data_seed)
    
    # Initialize psutil CPU monitoring
    psutil.cpu_percent()  # First call to initialize

    # Initialize W&B (only on master process) using shared utils; construct name if absent
    if master_process:
        effective_batch_size = config.batch_size * config.grad_acc_steps * (world_size if ddp else 1)
        if getattr(config, 'wandb_name', None) is None and getattr(config, 'use_wandb', False):
            model_suffix = "_rope" if not config.no_rope else ("_abacus" if config.digits > 1 else "")
            control_bits_str = config.control_bits.replace(',', '_') if getattr(config, 'type', None) not in ['LCG', 'TLCG'] else None
            type_name = '-'.join(config.type_list) if hasattr(config, 'type_list') else config.type
            cb_segment = f"_cb{control_bits_str}" if control_bits_str is not None else ""
            config.wandb_name = f"{type_name}_m{config.m}{cb_segment}_kp{config.bits_to_keep}_vs{config.vocab_size}_sl{config.seq_len}_b{config.base}_nd{config.digits}_na{config.n_a}_nc{config.n_c}_ne{config.n_example}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}{model_suffix}_dI{config.data_seed}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Twarm{config.warm_steps}_T{config.num_steps}_B{effective_batch_size}_wd{config.weight_decay}"
    init_wandb(config, master_process)

    # Load excluded AC values if specified
    excluded_a, excluded_c = load_excluded_ac_values(config.exclude_ac_paths, master_process)

    # Generate data using lowmem method
    t0 = time.time()  # Start timing here
    train_dataset, test_dataset, train_a, train_c, test_a, test_c, per_type_test_datasets = generate_lowmem_data(config, rng, master_process, excluded_a, excluded_c)

    # Set up data loaders
    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank,
                                                                    shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                   num_replicas=world_size,
                                                                   rank=rank,
                                                                   shuffle=False)
        train_loader = DataLoader(train_dataset,
                             batch_size=config.batch_size,
                             sampler=train_sampler,
                             num_workers=config.num_workers,
                             pin_memory=True,
                             persistent_workers=True,
                             prefetch_factor=4,
                             drop_last=True)
        test_loader = DataLoader(test_dataset,
                            batch_size=config.batch_size,
                            sampler=test_sampler,
                            num_workers=config.num_workers,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2,
                            drop_last=True)
    else:
        # Use configurable number of workers for server performance
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                                 num_workers=config.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                                num_workers=config.num_workers, pin_memory=True, drop_last=True)
    t1 = time.time()
    if master_process: 
        print(f"train data size: {len(train_dataset)}, test data size: {len(test_dataset)}")
        print(f"data generation took: {t1-t0:.2f} seconds")
        print(f"using {config.num_workers} DataLoader workers")
        print("config: ",config)

    # Setup per-type evaluation using the datasets returned from generate_lowmem_data
    per_test_loaders = {}
    per_type_labels = []
    per_type_eval_results = []
    
    if per_type_test_datasets:
        if master_process:
            print("Setting up per-type evaluation with lowmem datasets...")
            print(f"Available per-type datasets: {list(per_type_test_datasets.keys())}")
        
        # Helper to create a DataLoader from a dataset with optional distributed sampler
        def _make_loader(ds):
            bs = max(1, min(int(config.batch_size), len(ds)))
            if ddp:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
                )
                return DataLoader(ds, batch_size=bs, sampler=sampler, pin_memory=True,
                                  num_workers=0, drop_last=False)
            else:
                return DataLoader(ds, batch_size=bs, shuffle=False, pin_memory=True,
                                  num_workers=0, drop_last=False)
        
        # Create loaders for each per-type test dataset
        for type_label, type_dataset in per_type_test_datasets.items():
            per_test_loaders[type_label] = _make_loader(type_dataset)
        
        # Establish stable order for saving/logging
        per_type_labels = sorted(per_test_loaders.keys())
        
        if master_process:
            print(f"Created per-type test loaders for: {per_type_labels}")
            for label in per_type_labels:
                loader = per_test_loaders[label]
                dataset_size = len(loader.dataset) if hasattr(loader, 'dataset') else "unknown"
                print(f"  - {label}: {dataset_size} sequences")
    elif master_process:
        print("Skipping per-type evaluation for single-type configuration (memory efficient)")

    # Initialize model
    if not config.no_rope:
        model = GPT_RoPE(GPTConfig(block_size=config.context_len,n_embd=config.n_embd,n_head=config.n_head,
                              vocab_size=config.vocab_size, n_layer=config.n_layer))
        if master_process:
            print("Using GPT model with RoPE positional embedding")
    else:
        # Use abacus embedding for multi-digit representations
        if config.digits > 1:
            model = GPT_oth_abacus(GPTConfig_abacus(block_size=config.context_len,n_embd=config.n_embd,n_head=config.n_head,
                                  vocab_size=config.vocab_size, n_layer=config.n_layer, digits=config.digits))
            if master_process:
                print(f"Using GPT model with abacus embedding (digits={config.digits})")
        else:
            model = GPT(GPTConfig(block_size=config.context_len,n_embd=config.n_embd,n_head=config.n_head,
                                  vocab_size=config.vocab_size, n_layer=config.n_layer))
            if master_process:
                print("Using standard GPT model")

    model.to(device)

    # Model compilation and setup
    if hasattr(torch, '_dynamo'):
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.suppress_errors = True

    # Print model architecture if master process
    if master_process:
        print("="*80)
        print("MODEL ARCHITECTURE:")
        print("-"*80)
        
        # Get model summary in a more readable format
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Print model structure with better formatting
        print(model)
        print("-"*80)
        
        # Print parameter counts by layer
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Print model configuration summary
        print("-"*80)
        print(f"Model configuration:")
        print(f"  - Layers:           {config.n_layer}")
        print(f"  - Heads:            {config.n_head}")
        print(f"  - Embedding dim:    {config.n_embd}")
        print(f"  - Vocabulary size:  {config.vocab_size}")
        print(f"  - Context length:   {config.context_len}")
        print("="*80)
        print("MEMORY EFFICIENCY:")
        print(f"  - Using on-demand sequence generation")
        print(f"  - No caching - pure memory efficiency")
        print(f"  - Configurable DataLoader workers: {config.num_workers}")
        print("="*80)
        
        # Log model info to W&B summary (not as charts)
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.summary.update({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/parameters_M": total_params / 1e6,  # In millions
                "data/generation_time_seconds": t1-t0,
                "data/train_dataset_size": len(train_dataset),
                "data/test_dataset_size": len(test_dataset),
            })

    # Compile if supported
    try:
        if hasattr(torch, 'compile') and (sys.version_info < (3, 12)):
            model = torch.compile(model)
        else:
            if master_process:
                print("Skipping torch.compile (unsupported on this Python version)")
    except Exception as compile_err:
        if master_process:
            print(f"Skipping torch.compile due to error: {compile_err}")
    
    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            static_graph=True,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # Set up optimizer and scheduler
    optimizer = raw_model.configure_optimizers(weight_decay=config.weight_decay, 
                                              learning_rate=config.lr_trgt, 
                                              beta1=config.beta1, 
                                              beta2=config.beta2, 
                                              device=device)

    # Create warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=config.warm_steps
    )

    # Create cosine annealing scheduler
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_steps - config.warm_steps,
        eta_min=config.lr_min
    )

    # Combine schedulers
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config.warm_steps]
    )

    # Calculate effective batch size and batches per epoch
    effective_batch_size = config.batch_size * config.grad_acc_steps * (world_size if ddp else 1)
    batches_per_epoch = len(train_loader)  # This already accounts for drop_last=True
    # Each gradient update represents grad_acc_steps mini-batches
    gradient_updates_per_epoch = batches_per_epoch // config.grad_acc_steps
    num_epoch = math.ceil(config.num_steps / gradient_updates_per_epoch)

    if master_process: 
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Number of batches per epoch: {batches_per_epoch}")
        print(f"Number of gradient updates per epoch: {gradient_updates_per_epoch}")
        print(f"Number of epochs to run: {num_epoch}")
        print("-"*80)
        print("Starting training...")
        t0 = time.time()

    # Save initial checkpoint if requested
    if master_process and config.save_checkpoints:
        eval_results = list()  # Initialize eval_results before saving initial checkpoint
        save_checkpoint(raw_model, optimizer, scheduler, 0, eval_results, config, world_size, ddp)

    # Wrap the main training code in a try-except block
    try:
        # Train the model
        if 'eval_results' not in locals():  # Only initialize if not already done above
            eval_results = list()
        # Set t1 to current time for timing calculations
        t1 = time.time()
        train(model, optimizer, scheduler, train_loader, test_loader, num_epoch, eval_results, device, config, master_process, device_type, ddp, rank, world_size, raw_model, t0, t1, effective_batch_size, train_a, train_c, test_a, test_c, train_dataset, test_dataset, per_test_loaders, per_type_labels, per_type_eval_results)

        # Update t1 after training is complete
        t1 = time.time()

        # Save final checkpoint if requested
        if master_process and config.save_checkpoints:
            # Get the final step count from eval_results or use config.num_steps
            final_step = eval_results[-1][0] if eval_results else config.num_steps
            save_checkpoint(raw_model, optimizer, scheduler, final_step, eval_results, config, world_size, ddp)
            print(f"Final checkpoint saved at step {final_step}")

        # Save results and clean up
        if master_process:
            print("="*80)
            print(f"TRAINING COMPLETE:")
            print(f"  - Steps: {config.num_steps}")
            print(f"  - Time: {t1-t0:.2f} seconds")
            print(f"  - Average time per step: {(t1-t0)/config.num_steps:.4f} seconds")
            if config.save_checkpoints:
                print(f"  - Checkpoints saved every {config.checkpoint_interval} steps")
            print("="*80)
            
            # Log final training metrics to W&B summary (not as charts)
            if config.use_wandb and WANDB_AVAILABLE:
                wandb.summary.update({
                    "training/total_time_seconds": t1-t0,
                    "training/avg_time_per_step": (t1-t0)/config.num_steps,
                    "training/steps_per_second": config.num_steps/(t1-t0),
                    "training/final_step": eval_results[-1][0] if eval_results else config.num_steps,
                    "learning_curves/final_train_loss": eval_results[-1][1] if eval_results else float('nan'),
                    "learning_curves/final_test_loss": eval_results[-1][2] if eval_results else float('nan'),
                    "learning_curves/final_train_acc": eval_results[-1][3] if eval_results else float('nan'),
                    "learning_curves/final_test_acc": eval_results[-1][4] if eval_results else float('nan'),
                })
            
            # Save all results (only on master process)
            if master_process:
                # Pass per-type loaders if available (for lowmem per-type evaluation)
                per_type_loaders_for_save = per_test_loaders if per_test_loaders else None
                per_type_labels_for_save = per_type_labels if per_type_labels else None
                save_results(config, eval_results, raw_model, effective_batch_size, train_a, train_c, test_a, test_c, train_dataset, test_dataset, wandb_available=WANDB_AVAILABLE, per_type_loaders=per_type_loaders_for_save, per_type_labels=per_type_labels_for_save)
                
                # Save training a,c values separately if requested
                if getattr(config, 'save_params', False):
                    from utils.curriculum_utils import get_training_ac_values_filename
                    ac_filename = get_training_ac_values_filename(config)
                    ac_path = os.path.join(config.results_dir, f"{ac_filename}.pt")
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(ac_path), exist_ok=True)

                    training_ac_values = {
                        'train_a': train_a,
                        'train_c': train_c
                    }

                    torch.save({
                        'training_ac_values': training_ac_values,
                        'config': { # Simplified config for a,c values file
                            'type': config.type,
                            'm': config.m,
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
                # Save per-type evaluation CSV if available
                if per_type_labels and per_type_eval_results:
                    # Build columns: step, then per-label triplets
                    columns = ['step']
                    for label in per_type_labels:
                        columns.extend([f'{label}_loss', f'{label}_acc', f'{label}_last_acc'])
                    df = pd.DataFrame(per_type_eval_results, columns=columns)
                    # Derive base filename similarly to eval file naming
                    base_path = create_base_path(config, effective_batch_size)
                    per_type_path = f"{config.results_dir}/per_type_eval_{base_path}.csv"
                    per_type_path = get_unique_filename(per_type_path)
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(per_type_path), exist_ok=True)
                    df.to_csv(per_type_path, index=False)
                    print(f"Per-type evaluation results saved to: {per_type_path}")
                
                # Finish W&B run
                if config.use_wandb and WANDB_AVAILABLE:
                    finish_wandb(config)

    except Exception as e:
        # Print the error for debugging
        if master_process:
            print(f"Error occurred during training: {e}")
            import traceback
            traceback.print_exc()
        
        # Clean up distributed process group before exiting
        if ddp:
            try:
                dist.destroy_process_group()
                if master_process:
                    print("Distributed process group cleaned up after error.")
            except Exception as cleanup_error:
                if master_process:
                    print(f"Error during cleanup: {cleanup_error}")
        
        # Finish W&B run even on error
        if config.use_wandb and master_process and WANDB_AVAILABLE:
            finish_wandb(config)
        
        # Re-raise the exception to maintain error behavior
        raise

    # Normal cleanup (this will only run if no exception occurred)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
