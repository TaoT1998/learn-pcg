"""
Utility functions for curriculum learning in PRNG tasks.

This module provides functions for:
1. Phase-based Learning Rate Scheduling:
   - Each phase can have its own warmup steps and duration
   - Supports linear warmup, constant LR, and cosine decay
   - Default schedule: 10% warmup, 90% training with cosine decay

2. Multi-Modulus Curriculum:
   - Handles multiple moduli (difficulty levels) simultaneously
   - Weighted sampling based on curriculum phase
   - Flexible phase transitions (linear, cosine, step, exp)
   - Per-phase modulus weights

3. Evaluation and Results:
   - Multi-modulus evaluation
   - Per-modulus metrics tracking
   - Results saving and visualization
   - Optional WandB integration

4. Data Management:
   - Weighted sampling for curriculum learning
   - Multi-dataset concatenation and sampling
   - Distributed training support
"""

import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import torch.distributed as dist
import math
import numpy as np
import pandas as pd
import os

from .file_utils import get_unique_filename
from .training_utils import evaluate_model

# Import W&B utilities
from utils.wandb_utils import log_correctness_plot


def parse_moduli_list(moduli_str):
    """Parse a comma-separated string of moduli into a sorted list of integers
    
    Args:
        moduli_str: String like "256,1024,4096", single integer, or list with Python expressions
        
    Returns:
        List of integers sorted in ascending order
    """
    if isinstance(moduli_str, (list, tuple)):
        moduli = []
        for m in moduli_str:
            if isinstance(m, str) and ('**' in m or '*' in m):
                # Evaluate Python expressions like '2**16'
                try:
                    moduli.append(eval(m))
                except:
                    moduli.append(int(m))
            else:
                moduli.append(int(m))
    elif isinstance(moduli_str, int):
        moduli = [moduli_str]
    else:
        # Parse comma-separated string
        moduli = []
        for m in str(moduli_str).split(','):
            m = m.strip()
            if '**' in m or '*' in m:
                # Evaluate Python expressions
                try:
                    moduli.append(eval(m))
                except:
                    moduli.append(int(m))
            else:
                moduli.append(int(m))
    
    # Sort in ascending order (easy to hard)
    moduli.sort()
    return moduli


def get_multi_alpha(step, curriculum_config):
    """Calculate alpha weights for multiple moduli based on curriculum configuration
    
    Args:
        step: Current training step
        curriculum_config: Dictionary with curriculum configuration
            {
                'moduli': [256, 1024, 4096, 16384],  # List of moduli in increasing difficulty
                'phases': [
                    {
                        'phase_steps': 1000,
                        'transition_steps': 200,
                        'start_weights': [1.0, 0.0, 0.0, 0.0],
                        'end_weights': [0.8, 0.2, 0.0, 0.0],
                        'transition': 'linear'
                    },
                    # ... more phases
                ]
            }
    
    Returns:
        List of alpha weights for each modulus, sums to 1.0
    """
    moduli = curriculum_config['moduli']
    phases = curriculum_config['phases']
    
    # Find current phase
    cumulative_steps = 0
    current_phase = None
    phase_start_step = 0
    
    for i, phase in enumerate(phases):
        phase_steps = phase.get('phase_steps', 0)
        if phase_steps == -1:  # Rest of training
            current_phase = phase
            phase_start_step = cumulative_steps
            break
        elif step < cumulative_steps + phase_steps:
            current_phase = phase
            phase_start_step = cumulative_steps
            break
        cumulative_steps += phase_steps
    
    if current_phase is None:
        # Fallback to last phase if step exceeds all defined phases
        current_phase = phases[-1]
        phase_start_step = cumulative_steps - current_phase.get('phase_steps', 0)
    
    # Get phase configuration
    transition_steps = current_phase.get('transition_steps', 0)
    start_weights = current_phase.get('start_weights', [])
    end_weights = current_phase.get('end_weights', [])
    transition_type = current_phase.get('transition', 'linear')
    
    # Calculate step within current phase
    step_in_phase = step - phase_start_step
    
    # Determine if we're in transition period or constant period
    if step_in_phase < transition_steps and transition_steps > 0:
        # We're in the transition period - interpolate between start and end weights
        progress = step_in_phase / transition_steps
        
        # Apply transition function
        if transition_type == 'linear':
            transition_progress = progress
        elif transition_type == 'cosine':
            transition_progress = 0.5 * (1 - math.cos(math.pi * progress))
        elif transition_type == 'exp':
            transition_progress = 1 - math.exp(-3 * progress)
        elif transition_type == 'step':
            transition_progress = 1.0 if progress > 0.5 else 0.0
        else:
            transition_progress = progress  # Default to linear
        
        # Interpolate weights
        weights = []
        for start_w, end_w in zip(start_weights, end_weights):
            w = start_w + transition_progress * (end_w - start_w)
            weights.append(w)
        
        # Normalize to sum to 1.0
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            # Fallback to end weights if all weights are zero
            weights = end_weights
        
    else:
        # We're in the constant period - use end weights
        weights = end_weights
    
    return weights




def create_curriculum_config_custom(moduli, phases_config, transition_type='linear'):
    """Create a custom curriculum configuration
    
    Args:
        moduli: List of moduli in increasing difficulty order
        phases_config: List of phase configurations, each with 'steps' and 'weights'
        transition_type: Type of transition between phases
        
    Returns:
        Curriculum configuration dictionary
    """
    return {
        'moduli': moduli,
        'phases': phases_config,
        'transition_type': transition_type
    }


def create_multi_weighted_sampler(datasets, weights, rank=None, world_size=None, seed=None, epoch=0):
    """Create a weighted sampler for multiple datasets in curriculum training
    
    Args:
        datasets: List of datasets
        weights: List of weights for each dataset (should sum to 1.0)
        rank: Process rank for DDP
        world_size: Number of processes for DDP
        seed: Random seed
        epoch: Current epoch for seeding
        
    Returns:
        WeightedRandomSampler
    """
    # Calculate weights for each sample across all datasets
    total_samples = sum(len(dataset) for dataset in datasets)
    sample_weights = torch.zeros(total_samples)
    
    start_idx = 0
    for i, (dataset, weight) in enumerate(zip(datasets, weights)):
        end_idx = start_idx + len(dataset)
        sample_weights[start_idx:end_idx] = weight
        start_idx = end_idx
    
    if rank is not None and world_size is not None:
        # For DDP: keep weights length equal to dataset length and zero out non-local positions
        mask = torch.zeros(total_samples, dtype=torch.bool)
        mask[rank::world_size] = True
        rank_weights = sample_weights.clone()
        rank_weights[~mask] = 0.0
        num_samples = int(mask.sum().item())
        # Set seed for this epoch and rank
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed + epoch * world_size + rank)
            sampler = WeightedRandomSampler(rank_weights, num_samples=num_samples, replacement=True, generator=g)
        else:
            sampler = WeightedRandomSampler(rank_weights, num_samples=num_samples, replacement=True)
    else:
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed + epoch)
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=g)
        else:
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    return sampler


def evaluate_multi_curriculum_model(model, test_loaders, device, device_type, config, save_correctness=None, rank=None, world_size=None, master_process=True, train_datasets=None, train_batch=None, train_correct_vector=None):
    """Evaluate model on multiple test sets with different moduli
    
    Args:
        model: Model to evaluate
        test_loaders: List of test DataLoaders for each modulus
        device: Device to run evaluation on
        device_type: Device type for autocast
        config: Configuration object
        save_correctness: Whether to save correctness data. If None, uses config.save_correctness
        rank: Rank in distributed training (None for single process)
        world_size: World size in distributed training (None for single process)
        master_process: Whether this is the master process
        train_datasets: List of training datasets for each modulus (needed for legacy correctness analysis)
        train_batch: Optional tuple (x, y) from the last training batch to compute global train correctness cheaply
    
    Returns:
        Dictionary of test metrics for each modulus
    """
    results = {}
    correctness_data = {}
    
    # Get moduli from curriculum config
    moduli = config.curriculum_config['moduli']
    
    # Lightweight global train correctness: prefer provided vector; otherwise optional batch compute
    should_save_global = save_correctness if save_correctness is not None else config.save_correctness
    try:
        reduced_train_correct = None
        if train_correct_vector is not None:
            # Convert to tensor and reduce across ranks if needed
            tc = torch.tensor(train_correct_vector, device=device, dtype=torch.float32)
            if rank is not None:
                torch.distributed.all_reduce(tc, op=torch.distributed.ReduceOp.SUM)
                tc = tc / float(world_size)
            reduced_train_correct = tc.detach().cpu().numpy()
        elif train_batch is not None and master_process:
            # Fallback: compute from batch on master only
            x_batch, y_batch = train_batch
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            if device_type == 'cuda':
                autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            elif device_type == 'cpu':
                autocast_ctx = torch.autocast(device_type='cpu', dtype=torch.float16)
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()
            with torch.no_grad(), autocast_ctx:
                logits, _ = model(x_batch, y_batch)
            preds = logits.argmax(dim=-1)
            digits = int(getattr(config, 'digits', 1))
            truth_adj = y_batch[:, digits-1:]
            preds_adj = preds[:, digits-1:]
            token_correct = (truth_adj == preds_adj)
            number_correct = token_correct[:, ::digits].clone()
            for d in range(1, digits):
                number_correct &= token_correct[:, d::digits]
            reduced_train_correct = number_correct.float().mean(dim=0).cpu().numpy()
        # Save only on master
        if should_save_global and master_process and reduced_train_correct is not None:
            correctness_data['train_correct'] = reduced_train_correct
    except Exception as e:
        print(f"Warning: Could not prepare train_correct vector: {e}")

    # Evaluate on each test set
    for i, test_loader in enumerate(test_loaders):
        modulus = moduli[i]
        
        # Get test metrics
        test_loss, test_acc, test_last_acc = evaluate_model(
            model, test_loader, device, device_type, config
        )
        
        # Store results
        results[f'm{modulus}'] = {
            'loss': test_loss,
            'acc': test_acc,
            'last_acc': test_last_acc
        }
        
        # Defer distributed aggregation to caller to avoid double-reduction
        
        # Get correctness data if requested (only on master process in distributed training)
        should_save = save_correctness if save_correctness is not None else config.save_correctness
        if should_save and master_process:
            from utils.eval import get_predictions
            # In distributed training, we'll only do this on the master process
            if rank is None or rank == 0:
                # Compute TEST correctness over the full test dataset for this modulus
                test_dataset = getattr(test_loader, 'dataset', None)
                if test_dataset is not None:
                    try:
                        test_truth, test_predictions = get_predictions(model=model, dataset=test_dataset, batch_size=config.batch_size)
                        digits = int(getattr(config, 'digits', 1))
                        truth_adj = test_truth[:, digits-1:]
                        preds_adj = test_predictions[:, digits-1:]
                        token_correct = (truth_adj == preds_adj)
                        number_correct = token_correct[:, ::digits].copy()
                        for d in range(1, digits):
                            number_correct &= token_correct[:, d::digits]
                        test_correct_avg = number_correct.mean(axis=0)
                        # Store TEST correctness data for this modulus
                        correctness_data[f'test_m{modulus}'] = test_correct_avg
                    except Exception as e:
                        print(f"Warning: Could not compute test correctness for m={modulus}: {e}")
    
    # Save correctness data if requested (only on master process)
    if correctness_data and (save_correctness if save_correctness is not None else config.save_correctness) and master_process and (rank is None or rank == 0):
        import os
        import numpy as np
        from utils.file_utils import get_unique_filename
        

        # Create base filename
        base_path = get_multi_curriculum_base_filename(config)
        path = os.path.join(config.results_dir, f"correctness_{base_path}.npz")
        path = get_unique_filename(path)
        
        # Save correctness data for all moduli
        np.savez(path, **correctness_data)
        print(f"Multi-curriculum correctness data saved to: {path}")
        
        # Log correctness data to W&B if available
        if config.use_wandb:
            try:
                import matplotlib.pyplot as plt
                
                # Create a clear line plot showing accuracy vs sequence position for each modulus
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                # Determine whether to plot test or train per-modulus curves
                moduli = config.curriculum_config['moduli']
                # Prefer test curves if present
                first_key = None
                for m in moduli:
                    if f'test_m{m}' in correctness_data:
                        first_key = f'test_m{moduli[0]}'
                        break
                if first_key is None:
                    # Fallback to train curves
                    for m in moduli:
                        if f'train_m{m}' in correctness_data:
                            first_key = f'train_m{moduli[0]}'
                            break
                if first_key is None:
                    raise ValueError("No per-modulus correctness arrays found to plot")

                positions = np.arange(len(correctness_data[first_key]))
                label_prefix = 'test' if first_key.startswith('test_') else 'train'
                for m in moduli:
                    key = f'{label_prefix}_m{m}'
                    if key in correctness_data:
                        acc_data = correctness_data[key]
                        ax.plot(positions, acc_data, '-', label=f'{label_prefix} m={m}', linewidth=2, marker='o', markersize=4)
                
                ax.set_xlabel('Sequence Position')
                ax.set_ylabel('Accuracy')
                ax.set_title('Final Model: Accuracy by Sequence Position (All Moduli)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                
                # Create table data
                table_data = []
                for pos in positions:
                    row = [int(pos)]
                    for m in moduli:
                        key = f'{label_prefix}_m{m}'
                        if key in correctness_data:
                            row.append(f"{float(correctness_data[key][pos]):.3f}")
                        else:
                            row.append("")
                    table_data.append(row)
                
                columns = ["Position"] + [f"{label_prefix}_m{m}_Acc" for m in moduli]
                
                # Log to W&B using utility function
                log_correctness_plot(config, fig, table_data, columns)
                plt.close(fig)
                
            except Exception as e:
                print(f"Error logging correctness to W&B: {e}")
    
    return results


def get_multi_curriculum_base_filename(config):
    """Create a base filename with all relevant multi-curriculum parameters"""
    # Model structure
    model_str = f"n{config.n_embd}_h{config.n_head}_d{config.n_layer}"
    
    # Data parameters
    data_str = f"na{config.n_a}_nc{config.n_c}_ne{config.n_example}_vs{config.vocab_size}_sl{config.seq_len}"
    # Only include control bits in filename when relevant (exclude LCG and TLCG)
    if getattr(config, 'type', None) not in ['LCG', 'TLCG']:
        if hasattr(config, 'control_bits') and config.control_bits:
            data_str += f"_cb{config.control_bits}"
    
    # Training settings - use total steps since phase details are in curriculum part
    total_steps = config.num_steps
    if hasattr(config, 'custom_phases') and config.custom_phases:
        # Calculate total steps from all phases
        total_steps = sum(phase.get('phase_steps', 0) for phase in config.custom_phases)
    
    # Effective batch size per optimizer update: per-device batch × grad_acc_steps × world_size
    grad_acc = getattr(config, 'grad_acc_steps', 1)
    world_size = getattr(config, 'world_size', 1)
    effective_batch = int(config.batch_size) * int(grad_acc) * int(world_size)
    train_str = f"T{total_steps}_B{effective_batch}_lr{config.lr_trgt}_wd{config.weight_decay}"
    
    # Random seeds
    seed_str = f"dI{config.data_seed}_I{config.main_seed}"
    
    # Curriculum parameters
    moduli = config.curriculum_config['moduli']
    moduli_str = '_'.join(map(str, moduli))
    
    # Get transition type from first phase (same for all phases)
    phases = config.curriculum_config.get('phases', [])
    if phases and 'transition' in phases[0]:
        transition_type = phases[0]['transition']
    else:
        # Fallback to global transition_type or default
        transition_type = config.curriculum_config.get('transition_type', 'linear')
    
    curriculum_str = f"moduli{moduli_str}_trans{transition_type}"
    
    # Add alpha weights information from curriculum phases
    alpha_weights_str = ""
    if hasattr(config, 'custom_phases') and config.custom_phases:
        # Create a compact representation of the complete curriculum schedule
        curriculum_parts = []
        for i, phase in enumerate(config.custom_phases):
            if 'start_weights' in phase and 'end_weights' in phase:
                start_weights = phase['start_weights']
                end_weights = phase['end_weights']
                # Format: p1s1000u100t200a1.0,0.0->0.8,0.2 for phase 1
                phase_steps = phase.get('phase_steps', 0)
                warmup_steps = phase.get('warmup_steps', 0)
                transition_steps = phase.get('transition_steps', 0)
                
                # Format weights like lr is formatted (default Python float string)
                start_str = ','.join([str(float(w)) for w in start_weights])
                end_str = ','.join([str(float(w)) for w in end_weights])
                # Encode LR decay type per phase: const or cos
                lr_decay_val = str(phase.get('lr_decay', 'constant')).lower()
                lr_token = 'cos' if lr_decay_val == 'cosine' else 'const'
                
                # Include 'd{lr}' token between t and a to capture LR annealing per phase
                curriculum_parts.append(
                    f"p{i+1}s{phase_steps}u{warmup_steps}t{transition_steps}d{lr_token}a{start_str}-{end_str}"
                )
        
        if curriculum_parts:
            # Join all curriculum parts into a single string
            alpha_weights_str = f"_c{'_'.join(curriculum_parts)}"
    
    # Add pretrain info if available
    pretrain_str = ""
    if hasattr(config, 'pretrain_m'):
        if config.pretrain_m is not None:
            pretrain_str = f"prem{config.pretrain_m}_"
        else:
            pretrain_str = "premNone_"
    
    # Combine all parts
    base_filename = f"{config.type}_{pretrain_str}{curriculum_str}{alpha_weights_str}_{model_str}_{data_str}_{train_str}_{seed_str}"
    
    return base_filename


def get_training_ac_values_filename(config):
    """Create a filename for training a,c values that only depends on data generation parameters"""
    # Data parameters - include test parameters since they affect training values
    data_str = f"na{config.n_a}_nc{config.n_c}_nta{config.n_test_a}_ntc{config.n_test_c}_ne{config.n_example}_vs{config.vocab_size}_sl{config.seq_len}"
    
    # Random seeds
    seed_str = f"dI{config.data_seed}_I{config.main_seed}"
    
    # Handle both single-modulus and multi-modulus cases
    if hasattr(config, 'curriculum_config') and 'moduli' in config.curriculum_config:
        # Multi-modulus curriculum case
        moduli = config.curriculum_config['moduli']
        moduli_str = '_'.join(map(str, moduli))
        filename = f"training_ac_values_moduli{moduli_str}_{data_str}_{seed_str}"
    else:
        # Single-modulus case
        filename = f"training_ac_values_m{config.m}_{data_str}_{seed_str}"
    
    return filename


def save_multi_curriculum_results(config, eval_results, output_dir):
    """Save multi-curriculum training evaluation results to CSV files
    
    Args:
        config: Configuration object
        eval_results: List of evaluation results
        output_dir: Directory to save results
    """
    if not eval_results:
        print("No evaluation results to save")
        return
    
    # Create filenames using base filename
    base_filename = get_multi_curriculum_base_filename(config)
    eval_filename = f"{base_filename}.csv"
    
    # Create column names
    moduli = config.curriculum_config['moduli']
    columns = ['step', 'lr', 'train_loss', 'train_acc', 'train_last_acc', 'grad_norm']
    
    # Add alpha weights columns
    for i, m in enumerate(moduli):
        columns.append(f'alpha_m{m}')
    
    # Add test metrics columns
    for m in moduli:
        columns.extend([f'test_loss_m{m}', f'test_acc_m{m}', f'test_last_acc_m{m}'])
    
    # Save evaluation results
    eval_path = get_unique_filename(os.path.join(output_dir, eval_filename))
    eval_df = pd.DataFrame(eval_results, columns=columns)
    eval_df.to_csv(eval_path, index=False)
    print(f"Saved multi-curriculum evaluation results to {eval_path}")
    
    return eval_path


def save_multi_curriculum_checkpoint(model, optimizer, scheduler, step, eval_results, config, world_size, ddp):
    """Save multi-curriculum model checkpoint using the multi-curriculum base filename."""
    import torch
    import os
    
    # Create checkpoint directory
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Create checkpoint data
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'eval_results': eval_results,
        'config': config
    }
    
    # Use multi-curriculum base filename
    base_filename = get_multi_curriculum_base_filename(config)
    checkpoint_path = f'{config.results_dir}/multi_curriculum_checkpoint_{base_filename}_step{step}.pt'
    checkpoint_path = get_unique_filename(checkpoint_path)
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Multi-curriculum checkpoint saved at step {step} to: {checkpoint_path}")
    
    return checkpoint_path


def print_multi_curriculum_info(config):
    """Print information about multi-curriculum training setup"""
    curriculum_config = config.curriculum_config
    moduli = curriculum_config['moduli']
    phases = curriculum_config['phases']
    transition_type = curriculum_config.get('transition_type', 'linear')
    
    print(f"Multi-Modulus Curriculum Training Setup ({transition_type} transitions):")
    print(f"  Moduli (easy -> hard): {moduli}")
    
    # Show bits_to_keep information if available
    if hasattr(config, 'moduli_bits_to_keep') and config.moduli_bits_to_keep is not None:
        print(f"  Custom bits_to_keep: {config.moduli_bits_to_keep}")
        print(f"  Moduli details:")
        for i, (m, btk) in enumerate(zip(moduli, config.moduli_bits_to_keep)):
            default_btk = int(math.ceil(math.log2(m))) if m > 0 else 0
            custom_str = " (custom)" if btk != default_btk else " (default)"
            print(f"    {i+1}. m={m}, bits_to_keep={btk}{custom_str}")
    else:
        print(f"  Using default bits_to_keep (log2(m)) for all moduli")
    
    print(f"  Number of phases: {len(phases)}")
    print()
    
    cumulative_steps = 0
    for i, phase in enumerate(phases):
        # Handle both legacy format ('steps') and new YAML format ('phase_steps')
        if 'steps' in phase:
            steps = phase['steps']
        elif 'phase_steps' in phase:
            steps = phase['phase_steps']
        else:
            steps = 0  # Default if neither key exists
            
        # Handle weights - could be 'weights' or start/end weights
        if 'weights' in phase:
            weights = phase['weights']
        elif 'start_weights' in phase and 'end_weights' in phase:
            weights = phase['start_weights']  # Show start weights
        else:
            weights = [0] * len(moduli)  # Default if no weights specified
        
        if steps == -1:
            steps_str = "rest of training"
        else:
            steps_str = f"{steps} steps"
            
        print(f"  Phase {i+1}: {steps_str}")
        for j, (m, w) in enumerate(zip(moduli, weights)):
            if w > 0:
                print(f"    m={m}: {w:.3f}")
        
        if steps != -1:
            cumulative_steps += steps
            print(f"    (cumulative: {cumulative_steps} steps)")
        print()


# Legacy functions for backwards compatibility
def get_alpha(step, initial_steps, cur_steps, alpha_start, alpha_end, schedule_type='linear'):
    """Legacy function for two-modulus curriculum - kept for backwards compatibility"""
    # Phase 1: Initial steps at alpha_start
    if step < initial_steps:
        return alpha_start
    
    # Phase 2: Curriculum transition
    curriculum_start = initial_steps
    curriculum_end = initial_steps + cur_steps
    
    if step < curriculum_end:
        # During curriculum phase
        progress = (step - curriculum_start) / cur_steps
        
        if schedule_type == 'linear':
            return alpha_start + (alpha_end - alpha_start) * progress
        elif schedule_type == 'cosine':
            cos_out = math.cos(math.pi * progress) * (-0.5) + 0.5
            return alpha_start + (alpha_end - alpha_start) * cos_out
        elif schedule_type == 'step':
            return alpha_end
        elif schedule_type == 'exp':
            decay = math.exp(-5 * progress)
            return alpha_end + (alpha_start - alpha_end) * decay
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    # Phase 3: After curriculum, stay at alpha_end
    return alpha_end


def create_weighted_sampler(dataset_easy, dataset_hard, alpha, rank=None, world_size=None, seed=None, epoch=0):
    """Legacy function for two-dataset curriculum - kept for backwards compatibility"""
    return create_multi_weighted_sampler([dataset_easy, dataset_hard], [alpha, 1-alpha], 
                                       rank, world_size, seed, epoch)


def setup_phase_based_schedulers(optimizer, config):
    """Set up learning rate schedulers based on curriculum phases with per-phase warmup
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration object with curriculum phases
        
    Returns:
        SequentialLR scheduler that handles per-phase warmup
    """
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be an instance of torch.optim.Optimizer")
    
    if not hasattr(config, 'num_steps') or config.num_steps <= 0:
        raise ValueError("config.num_steps must be positive")
    
    if hasattr(config, 'custom_phases') and config.custom_phases:
        # Use custom phases from config
        phases = config.custom_phases
    else:
        # Create a simple two-phase schedule: warmup then cosine decay
        phases = [
            {
                'name': 'Warmup',
                'phase_steps': int(config.num_steps * 0.1),  # 10% warmup by default
                'warmup_steps': int(config.num_steps * 0.1)
            },
            {
                'name': 'Training',
                'phase_steps': int(config.num_steps * 0.9),  # 90% training
                'warmup_steps': 0
            }
        ]
    
    schedulers = []
    milestones = []
    current_step = 0
    
    for i, phase in enumerate(phases):
        phase_steps = int(phase.get('phase_steps', 0))
        warmup_steps = int(phase.get('warmup_steps', 0))
        lr_decay_type = str(phase.get('lr_decay', 'constant')).lower()  # 'constant' or 'cosine'
        
        if phase_steps <= 0:
            continue
            
        # Create one scheduler per phase that handles both warmup and constant periods
        if warmup_steps > 0:
            # Phase with warmup: use LinearLR for warmup, then switch to constant or cosine per phase setting
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-4,  # Start from 0.01% of target LR
                end_factor=1.0,
                total_iters=warmup_steps
            )
            schedulers.append(warmup_scheduler)
            
            # Add milestone for transition from warmup to constant
            current_step += warmup_steps
            milestones.append(current_step)
            
            # Add post-warmup scheduler for remaining steps
            remaining_steps = phase_steps - warmup_steps
            if remaining_steps > 0:
                if lr_decay_type == 'cosine':
                    post_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=remaining_steps,
                        eta_min=getattr(config, 'lr_min', 1e-7)
                    )
                else:
                    post_scheduler = torch.optim.lr_scheduler.ConstantLR(
                        optimizer,
                        factor=1.0,
                        total_iters=remaining_steps
                    )
                schedulers.append(post_scheduler)
                current_step += remaining_steps
                if i < len(phases) - 1:  # Not the last phase
                    milestones.append(current_step)
        else:
            # No warmup: choose scheduler for entire phase based on lr_decay_type
            if lr_decay_type == 'cosine':
                phase_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=phase_steps,
                    eta_min=getattr(config, 'lr_min', 1e-7)
                )
            else:
                phase_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=1.0,
                    total_iters=phase_steps
                )
            schedulers.append(phase_scheduler)
            current_step += phase_steps
            if i < len(phases) - 1:  # Not the last phase
                milestones.append(current_step)
    
    # Add final cosine annealing if there are remaining steps
    if current_step < int(config.num_steps):
        final_steps = int(config.num_steps) - current_step
        final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=final_steps,
            eta_min=getattr(config, 'lr_min', 1e-7)
        )
        schedulers.append(final_scheduler)
        milestones.append(current_step)
    
    if len(schedulers) == 1:
        return schedulers[0]
    else:
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=schedulers,
            milestones=milestones
        )


# Legacy functions removed - only using multi-curriculum versions now