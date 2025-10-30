"""
W&B (Weights & Biases) utility functions for experiment tracking.

This module provides functions for initializing, logging, and managing W&B runs
for curriculum training experiments.
"""

import os
import argparse
import platform
import psutil
import torch

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception as e:
    print(f"Warning: wandb not available: {e}")
    WANDB_AVAILABLE = False


def init_wandb(config, master_process=True, world_size=1, ddp=False, device="cpu"):
    """
    Initialize W&B run for curriculum training.
    
    Args:
        config: Configuration object with W&B settings
        master_process: Whether this is the master process (for DDP)
        world_size: Number of processes for distributed training
        ddp: Whether using distributed data parallel
        device: Device being used for training
    
    Returns:
        bool: True if W&B was successfully initialized, False otherwise
    """
    if not config.use_wandb or not WANDB_AVAILABLE or not master_process:
        return False
    
    try:
        # Parse tags if provided
        tags = config.wandb_tags.split(',') if config.wandb_tags else None
        
        # Set W&B mode
        wandb_mode = "offline" if config.wandb_offline else "online"
        
        # Set W&B API key if provided
        if config.wandb_api_key:
            os.environ['WANDB_API_KEY'] = config.wandb_api_key
            print("W&B API key set from command line argument")
        
        # Calculate effective batch size for logging
        effective_batch_size = config.batch_size * config.grad_acc_steps * (world_size if ddp else 1)
        
        # Create comprehensive config dictionary
        wandb_config = {
            # Model hyperparameters
            "type": '+'.join(config.type_list) if hasattr(config, 'type_list') else config.type,
            "m": config.m,
            "control_bits": config.control_bits,
            "bits_to_keep": config.bits_to_keep,
            "vocab_size": config.vocab_size,
            "seq_len": config.seq_len,
            "base": config.base,
            "digits": config.digits,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "context_len": config.context_len,
            "no_rope": config.no_rope,
            
            # Data hyperparameters
            "n_a": config.n_a,
            "n_c": config.n_c,
            "n_test_a": config.n_test_a,
            "n_test_c": config.n_test_c,
            "n_example": config.n_example,
            "data_seed": config.data_seed,
            
            # Training hyperparameters
            "num_steps": config.num_steps,
            "warm_steps": config.warm_steps,
            "lr_trgt": config.lr_trgt,
            "lr_min": config.lr_min,
            "batch_size": config.batch_size,
            "grad_acc_steps": config.grad_acc_steps,
            "effective_batch_size": effective_batch_size,
            "weight_decay": config.weight_decay,
            "beta1": config.beta1,
            "beta2": config.beta2,
            "eval_interval": config.eval_interval,
            "main_seed": config.main_seed,
            
            # System info
            "world_size": world_size,
            "device": device,
            "ddp": ddp,
            "wandb_offline": config.wandb_offline,
            
            # Environment info
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "torch_cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        # Remove non-serializable items
        wandb_config.pop('_yaml_config', None)
        
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_name,
            tags=tags,
            notes=config.wandb_notes,
            mode=wandb_mode,
            config=wandb_config
        )
        
        print(f"W&B run initialized: {wandb.run.name}")
        if config.wandb_offline:
            print("W&B running in OFFLINE mode")
            print("To sync later, run: wandb sync <run_directory>")
            print("Find offline runs with: wandb offline")
        else:
            print(f"W&B URL: {wandb.run.url}")
        return True
        
    except Exception as e:
        print(f"Error initializing W&B: {e}")
        return False


def log_training_metrics(config, grad_updates, train_loss_avg, train_acc_avg, 
                        train_last_acc_avg, avg_grad_norm, current_lr, 
                        alpha_weights, moduli, test_metrics, device="cpu"):
    """
    Log training metrics to W&B.
    
    Args:
        config: Configuration object
        grad_updates: Current gradient update step
        train_loss_avg: Average training loss
        train_acc_avg: Average training accuracy
        train_last_acc_avg: Average training last token accuracy
        avg_grad_norm: Average gradient norm
        current_lr: Current learning rate
        alpha_weights: Current alpha weights for curriculum
        moduli: List of moduli
        test_metrics: Dictionary of test metrics per modulus
        device: Device being used for training
    """
    if not config.use_wandb or not WANDB_AVAILABLE:
        return
    
    try:
        log_dict = {
            "training/step": grad_updates,
            "learning_curves/train_loss": train_loss_avg,
            "learning_curves/train_acc": train_acc_avg,
            "learning_curves/train_last_acc": train_last_acc_avg,
            "training/grad_norm": avg_grad_norm,
            "training/learning_rate": current_lr
        }
        
        # Add alpha weights
        for m, w in zip(moduli, alpha_weights):
            log_dict[f"curriculum/alpha_m{m}"] = w
        
        # Add test metrics
        for m in moduli:
            log_dict[f"test_m{m}/loss"] = test_metrics[f'm{m}']['loss']
            log_dict[f"test_m{m}/acc"] = test_metrics[f'm{m}']['acc']
            log_dict[f"test_m{m}/last_acc"] = test_metrics[f'm{m}']['last_acc']
        
        # Add GPU memory info if using CUDA
        if device.startswith('cuda'):
            log_dict.update({
                "system/gpu_memory_allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
                "system/gpu_memory_reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
            })
        
        # Add CPU usage
        log_dict["system/cpu_percent"] = psutil.cpu_percent()
        
        wandb.log(log_dict, step=grad_updates)
        
    except Exception as e:
        print(f"Error logging to W&B: {e}")


def log_single_training_metrics(config, grad_updates, train_loss_avg, train_acc_avg, 
                               train_last_acc_avg, test_loss, test_acc, test_last_acc,
                               avg_grad_norm, current_lr, device="cpu"):
    """
    Log single training metrics to W&B (for non-curriculum training).
    
    Args:
        config: Configuration object
        grad_updates: Current gradient update step
        train_loss_avg: Average training loss
        train_acc_avg: Average training accuracy
        train_last_acc_avg: Average training last token accuracy
        test_loss: Test loss
        test_acc: Test accuracy
        test_last_acc: Test last token accuracy
        avg_grad_norm: Average gradient norm
        current_lr: Current learning rate
        device: Device being used for training
    """
    if not config.use_wandb or not WANDB_AVAILABLE:
        return
    
    try:
        log_dict = {
            "training/step": grad_updates,
            "learning_curves/train_loss": train_loss_avg,
            "learning_curves/test_loss": test_loss,
            "learning_curves/train_acc": train_acc_avg,
            "learning_curves/test_acc": test_acc,
            "learning_curves/train_last_acc": train_last_acc_avg,
            "learning_curves/test_last_acc": test_last_acc,
            "learning_curves/grad_norm": avg_grad_norm,
            "learning_curves/learning_rate": current_lr,
        }
        
        # Add GPU memory info if using CUDA
        if device.startswith('cuda'):
            log_dict.update({
                "system/gpu_memory_allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
                "system/gpu_memory_reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
            })
        
        # Add CPU usage
        log_dict["system/cpu_percent"] = psutil.cpu_percent()
        
        wandb.log(log_dict, step=grad_updates)
        
    except Exception as e:
        print(f"Error logging to W&B: {e}")


def log_correctness_plot(config, fig, table_data, columns):
    """
    Log correctness analysis plots and tables to W&B.
    
    Args:
        config: Configuration object
        fig: Matplotlib figure for correctness plot
        table_data: Data for correctness table
        columns: Column names for the table
    """
    if not config.use_wandb or not WANDB_AVAILABLE:
        return
    
    try:
        # Log the plot
        wandb.log({
            "final_correctness/plot": wandb.Image(fig),
        })
        
        # Create and log the table
        table = wandb.Table(data=table_data, columns=columns)
        wandb.log({"final_correctness/table": table})
        
        print("Final correctness analysis logged to W&B")
        
    except Exception as e:
        print(f"Error logging correctness to W&B: {e}")


def log_model_summary(config, total_params, trainable_params, generation_time, 
                     train_dataset_size, test_dataset_size):
    """
    Log model summary information to W&B.
    
    Args:
        config: Configuration object
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
        generation_time: Time taken for data generation
        train_dataset_size: Size of training dataset
        test_dataset_size: Size of test dataset
    """
    if not config.use_wandb or not WANDB_AVAILABLE:
        return
    
    try:
        wandb.summary.update({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/parameters_M": total_params / 1e6,  # In millions
            "data/generation_time_seconds": generation_time,
            "data/train_dataset_size": train_dataset_size,
            "data/test_dataset_size": test_dataset_size,
        })
        
    except Exception as e:
        print(f"Error logging model summary to W&B: {e}")


def log_final_training_metrics(config, total_time, num_steps, eval_results):
    """
    Log final training metrics to W&B summary.
    
    Args:
        config: Configuration object
        total_time: Total training time
        num_steps: Number of training steps
        eval_results: List of evaluation results
    """
    if not config.use_wandb or not WANDB_AVAILABLE:
        return
    
    try:
        final_step = eval_results[-1][0] if eval_results else num_steps
        
        wandb.summary.update({
            "training/total_time_seconds": total_time,
            "training/avg_time_per_step": total_time/num_steps,
            "training/steps_per_second": num_steps/total_time,
            "training/final_step": final_step,
            "learning_curves/final_train_loss": eval_results[-1][1] if eval_results else float('nan'),
            "learning_curves/final_test_loss": eval_results[-1][2] if eval_results else float('nan'),
            "learning_curves/final_train_acc": eval_results[-1][3] if eval_results else float('nan'),
            "learning_curves/final_test_acc": eval_results[-1][4] if eval_results else float('nan'),
        })
        
    except Exception as e:
        print(f"Error logging final metrics to W&B: {e}")


def finish_wandb(config):
    """
    Finish W&B run.
    
    Args:
        config: Configuration object
    """
    if config.use_wandb and WANDB_AVAILABLE:
        try:
            wandb.finish()
            print("W&B run finished")
        except Exception as e:
            print(f"Error finishing W&B run: {e}")


def add_wandb_args(parser):
    """
    Add W&B command line arguments to an argument parser.
    
    Args:
        parser: ArgumentParser object to add arguments to
    """
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='prng_scaling',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity (username or team)')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--wandb_tags', type=str, default=None,
                        help='W&B tags (comma-separated)')
    parser.add_argument('--wandb_notes', type=str, default=None,
                        help='W&B run notes')
    parser.add_argument('--wandb_offline', action='store_true',
                        help='Run W&B in offline mode')
    parser.add_argument('--wandb_api_key', type=str, default=None,
                        help='W&B API key (alternative to wandb login). Security: prefer WANDB_API_KEY env var')


def setup_wandb_config(config_dict, config):
    """
    Set up W&B configuration from YAML config.
    
    Args:
        config_dict: YAML configuration dictionary
        config: Configuration object to update
    """
    wandb_config = config_dict.get('wandb', {})
    config.use_wandb = wandb_config.get('use_wandb', False)
    config.wandb_project = wandb_config.get('project', 'prng_scaling')
    config.wandb_entity = wandb_config.get('entity', None)
    config.wandb_name = wandb_config.get('name', None)
    config.wandb_tags = ','.join(wandb_config.get('tags', [])) if wandb_config.get('tags') else None
    config.wandb_notes = wandb_config.get('notes', None)
    config.wandb_offline = wandb_config.get('offline', False)
    config.wandb_api_key = wandb_config.get('api_key', None) 