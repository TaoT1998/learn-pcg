import torch
import torch.distributed as dist
import numpy as np
import random
import os
import contextlib


def setup(rank, world_size):
    """Initialize the process group for distributed training."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def setup_random_seeds(seed):
    """Set up random seeds for reproducibility."""
    # Set seeds for all RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
    # Additional deterministic settings
    # Uncomment these for stricter determinism, but may cause performance degradation or errors
    
    # torch.use_deterministic_algorithms(True)  # Forces deterministic algorithms
    # torch.backends.cudnn.deterministic = True  # CuDNN deterministic mode  
    # torch.backends.cudnn.benchmark = False     # Disable CuDNN auto-tuning
    
    # Fill uninitialized memory for determinism (performance impact)
    # torch.utils.deterministic.fill_uninitialized_memory = True
    
    # Set CUDA environment variable for determinism (CUDA >= 10.2)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Note: For complete determinism with DataLoader workers, also need:
    # worker_init_fn = lambda worker_id: (torch.manual_seed(seed + worker_id), 
    #                                    np.random.seed(seed + worker_id))
    # when creating DataLoader with num_workers > 0


def evaluate_model(model, test_loader, device, device_type, config):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct_tokens = 0
    total_sequences = 0
    total_correct_last = 0
    
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device, non_blocking=True), y_test.to(device, non_blocking=True)
            # Guard autocast by device type
            if device_type == 'cuda':
                autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            elif device_type == 'cpu':
                autocast_ctx = torch.autocast(device_type='cpu', dtype=torch.float16)
            else:
                autocast_ctx = contextlib.nullcontext()
            with autocast_ctx:
                logits, loss = model(x_test, y_test) 
            batch_size = x_test.size(0)
            # Normalize loss per token for consistency with token accuracy
            total_loss += loss.item() * y_test.numel()
            preds = logits.argmax(dim=-1)
            total_correct_tokens += torch.sum(preds == y_test).item()
            total_tokens += y_test.numel()
            
            # Calculate last accuracy (check last `digits` tokens)
            total_correct_last += torch.sum(torch.all(preds[:,-config.digits:]==y_test[:,-config.digits:], dim=1)).item()
            total_sequences += batch_size
    
    # Avoid division by zero
    if total_tokens == 0 or total_sequences == 0:
        avg_loss = 0.0
        token_acc = 0.0
        last_acc = 0.0
    else:
        avg_loss = total_loss / total_tokens
        token_acc = total_correct_tokens / total_tokens
        last_acc = total_correct_last / total_sequences
    
    # Optional: print normalization summary once per call (caller handles logging)
    # print(f"Eval normalization: sequences={total_sequences}, tokens={total_tokens}")
    
    model.train()
    return avg_loss, token_acc, last_acc


def load_excluded_ac_values(exclude_ac_paths, master_process=True):
    """Load AC values from files that should be excluded from test set generation."""
    if not exclude_ac_paths:
        return set(), set()
    
    excluded_a = set()
    excluded_c = set()
    
    for path in exclude_ac_paths:
        if not os.path.exists(path):
            if master_process:
                print(f"Warning: Exclude AC path does not exist: {path}")
            continue
            
        if master_process:
            print(f"Loading excluded AC values from: {path}")
        
        try:
            if path.endswith('.npz'):
                # Load from numpy file
                data = np.load(path)
                if 'train_a' in data:
                    excluded_a.update(data['train_a'])
                if 'train_c' in data:
                    excluded_c.update(data['train_c'])
                if 'test_a' in data:
                    excluded_a.update(data['test_a'])
                if 'test_c' in data:
                    excluded_c.update(data['test_c'])
            elif path.endswith('.pt'):
                # Load from pytorch file
                data = torch.load(path, map_location='cpu')
                if isinstance(data, dict):
                    if 'training_ac_values' in data:
                        ac_data = data['training_ac_values']
                        if 'train_a' in ac_data:
                            excluded_a.update(ac_data['train_a'])
                        if 'train_c' in ac_data:
                            excluded_c.update(ac_data['train_c'])
                else:
                    # Assume it's a list of AC values
                    excluded_a.update(data)
            else:
                if master_process:
                    print(f"Warning: Unsupported file format: {path}")
                    
        except Exception as e:
            if master_process:
                print(f"Error loading {path}: {e}")
    
    if master_process:
        print(f"Loaded {len(excluded_a)} excluded a values and {len(excluded_c)} excluded c values")
    
    return excluded_a, excluded_c