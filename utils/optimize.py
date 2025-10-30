import math
import torch
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_lr(it,max_lr,min_lr,warmup_steps,max_steps):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def get_predictions(model, dataset, batch_size):
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    predictions = []
    truth = []
    with torch.no_grad():
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y) 
            preds = logits.argmax(dim=-1)
            predictions.append(preds)
            truth.append(y)
    predictions = torch.cat(predictions, dim=0)
    truth = torch.cat(truth, dim=0)
    return truth.cpu().numpy(), predictions.cpu().numpy()
    
        
