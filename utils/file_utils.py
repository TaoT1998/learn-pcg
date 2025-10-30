import os
import torch
import numpy as np
import pandas as pd
from .eval import get_predictions

def get_unique_filename(base_path):
    """Generate a unique filename by adding a numerical suffix if the file already exists.
    
    Args:
        base_path (str): The base file path to check and modify
        
    Returns:
        str: A unique filename, either the original if it doesn't exist,
             or with a numerical suffix added
    """
    if not os.path.exists(base_path):
        return base_path
    
    # Split the path into root and extension
    root, ext = os.path.splitext(base_path)
    counter = 2
    
    # Try with increasing counter until we find a filename that doesn't exist
    while os.path.exists(f"{root}_{counter}{ext}"):
        counter += 1
    
    return f"{root}_{counter}{ext}"


def create_base_path(config, effective_batch_size):
    """Create a standardized base path for all result files."""
    model_suffix = "_rope" if not config.no_rope else ("_abacus" if config.digits > 1 else "")
    # Only include control bits for PRNGs that use them (exclude LCG and TLCG)
    control_bits_str = config.control_bits.replace(',', '_') if getattr(config, 'type', None) not in ['LCG', 'TLCG'] else None
    cb_segment = f"_cb{control_bits_str}" if control_bits_str is not None else ""

    base_path = f"{config.type}_m{config.m}{cb_segment}_kp{config.bits_to_keep}_vs{config.vocab_size}_sl{config.seq_len}_b{config.base}_nd{config.digits}_na{config.n_a}_nc{config.n_c}_ne{config.n_example}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}{model_suffix}_dI{config.data_seed}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Twarm{config.warm_steps}_T{config.num_steps}_B{effective_batch_size}_wd{config.weight_decay}"
    
    return base_path


def save_results(config, eval_results, raw_model, effective_batch_size, train_a, train_c, test_a, test_c, train_dataset, test_dataset, wandb_available=False, per_type_loaders=None, per_type_labels=None):
    """Save all experiment results to files."""
    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    base_path = create_base_path(config, effective_batch_size)
    
    # Save evaluation results
    print(f"Saving evaluation results to: {config.results_dir}")
    df_eval = pd.DataFrame(eval_results, columns=['step', 'train_loss', 'test_loss', 'train_acc', 'test_acc', 'train_last_acc', 'test_last_acc', 'grad_norm'])
    path = f'{config.results_dir}/eval_{base_path}.tab'
    path = get_unique_filename(path)
    df_eval.to_csv(path, sep='\t')
    print(f"Evaluation results saved to: {path}")
    
    # Save model parameters if requested
    if config.save_params:
        path = f'{config.results_dir}/params_{base_path}.pth'
        path = get_unique_filename(path)
        
        # Get the state dict and remove the '_orig_mod.' prefix if it exists
        state_dict = raw_model.state_dict()
        clean_state_dict = {}
        for k, v in state_dict.items():
            # Remove '_orig_mod.' prefix if present
            if k.startswith('_orig_mod.'):
                clean_state_dict[k[10:]] = v  # Remove first 10 characters ('_orig_mod.')
            else:
                clean_state_dict[k] = v
        
        torch.save(clean_state_dict, path)
        print(f"Model parameters saved to: {path}")
        
        # Save training a and c values
        path = f'{config.results_dir}/trainac_{config.type}_m{config.m}_b{config.base}_d{config.digits}_na{config.n_a}_nc{config.n_c}_ne{config.n_example}_dI{config.data_seed}.npz'
        np.savez(path, train_a=train_a, train_c=train_c)
        print(f"Training a and c values saved to: {path}")
    
    # Save test a and c values if requested
    if config.save_test_values:
        path = f'{config.results_dir}/testac_{config.type}_m{config.m}_b{config.base}_d{config.digits}_na{config.n_a}_nc{config.n_c}_nta{len(test_a)}_ntc{len(test_c)}_dI{config.data_seed}.npz'
        
        # Check if file already exists
        if os.path.exists(path):
            print(f"Test a and c values file already exists at: {path}, skipping save")
        else:
            np.savez(path, test_a=test_a, test_c=test_c)
            print(f"Test a and c values saved to: {path}")
    
    # Save sequences if requested
    if config.save_correctness:
        from torch.utils.data import Subset
        
        raw_model.eval()
        
        # Get subset of training data for sequence analysis
        indices = torch.randperm(len(train_dataset))[:4096]
        subset_dataset = Subset(train_dataset, indices)
        train_truth, train_predictions = get_predictions(model=raw_model, dataset=subset_dataset, batch_size=config.batch_size)
        test_truth, test_predictions = get_predictions(model=raw_model, dataset=test_dataset, batch_size=config.batch_size)
        train_correct = train_truth == train_predictions
        test_correct = test_truth == test_predictions
        
        # Average correctness over sequences (per-sequence accuracy)
        train_correct_avg = train_correct.mean(axis=0)  # Average over sequence length dimension
        test_correct_avg = test_correct.mean(axis=0)    # Average over sequence length dimension
        
        # Prepare correctness data dictionary
        correctness_data = {
            'train_correct': train_correct_avg,
            'test_correct': test_correct_avg
        }
        
        # Add per-type correctness if available
        if per_type_loaders and per_type_labels:
            print("Computing per-type correctness...")
            for label in per_type_labels:
                if label in per_type_loaders:
                    loader = per_type_loaders[label]
                    try:
                        type_truth, type_predictions = get_predictions(model=raw_model, dataset=loader.dataset, batch_size=config.batch_size)
                        type_correct = type_truth == type_predictions
                        type_correct_avg = type_correct.mean(axis=0)  # Average over sequences
                        correctness_data[f'{label}_correct'] = type_correct_avg
                        print(f"  Computed correctness for {label}: shape {type_correct_avg.shape}")
                    except Exception as e:
                        print(f"  Warning: Could not compute correctness for {label}: {e}")
        
        path = f'{config.results_dir}/correctness_{base_path}.npz'
        path = get_unique_filename(path)
        np.savez(path, **correctness_data)
        print(f"Sequence correctness saved to: {path}")
        if per_type_loaders and per_type_labels:
            print(f"  Includes per-type correctness for: {list(per_type_labels)}")
        
        # Log correctness data to W&B if available
        if wandb_available and config.use_wandb:
            try:
                import wandb
                import matplotlib.pyplot as plt
                
                # Create a clear line plot showing accuracy vs sequence position
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                
                positions = np.arange(len(train_correct_avg))
                ax.plot(positions, train_correct_avg, 'b-', label='Train Accuracy', linewidth=2, marker='o')
                ax.plot(positions, test_correct_avg, 'r-', label='Test Accuracy', linewidth=2, marker='s')
                
                ax.set_xlabel('Sequence Position')
                ax.set_ylabel('Accuracy')
                ax.set_title('Final Model: Accuracy by Sequence Position')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                
                # Log the plot and table under final_correctness subfolder
                wandb.log({
                    "final_correctness/plot": wandb.Image(fig),
                })
                plt.close(fig)
                
                # Create a cleaner table for the data
                table_data = []
                for i, (train_acc, test_acc) in enumerate(zip(train_correct_avg, test_correct_avg)):
                    table_data.append([i, f"{float(train_acc):.3f}", f"{float(test_acc):.3f}"])
                
                table = wandb.Table(data=table_data, columns=["Position", "Train_Acc", "Test_Acc"])
                wandb.log({"final_correctness/table": table})
                
                print("Final correctness analysis logged to W&B")
                
            except Exception as e:
                print(f"Error logging correctness to W&B: {e}")
        
        raw_model.train()


def save_checkpoint(model, optimizer, scheduler, step, eval_results, config, world_size, ddp):
    """Save model checkpoint."""
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
    
    effective_batch_size = config.batch_size * config.grad_acc_steps * (world_size if ddp else 1)
    base_path = create_base_path(config, effective_batch_size)
    checkpoint_path = f'{config.results_dir}/checkpoint_{base_path}_step{step}.pt'
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at step {step} to: {checkpoint_path}") 