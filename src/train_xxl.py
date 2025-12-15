"""
XXL Training Script (Memory Optimized)
Features: Mixed Precision, Grad Accum, NumPy MemMap Data Loading
"""
import os
import time
import math
import json
import torch
import argparse
import numpy as np
import sys
from model import GPTConfig, GPT

# --- 1. CONFIGURATION ---
parser = argparse.ArgumentParser()
parser.add_argument('--n_layer', type=int, default=24) 
parser.add_argument('--n_embd', type=int, default=1600)
parser.add_argument('--max_iters', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=1) 
parser.add_argument('--grad_accum', type=int, default=32) 
args = parser.parse_args()

print(f"--- Starting XXL Training Run ({args.max_iters} steps) ---", flush=True)

n_head = args.n_embd // 64 
model_name = "xxl_final"
out_dir = f'checkpoints/{model_name}'
os.makedirs(out_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. LOAD DATA (MemMap Version) ---
data_dir = 'data'
with open(os.path.join(data_dir, 'vocab.json'), 'r') as f:
    vocab = json.load(f)
vocab_size = len(vocab)

class MemMapLoader:
    def __init__(self, split):
        path = os.path.join(data_dir, f'{split}.bin')
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        print(f" -> Loaded {split} data ({len(self.data)} tokens) via MemMap.", flush=True)
    
    def get_batch(self, batch_size, block_size):
        ix = torch.randint(len(self.data) - block_size, (batch_size,))
        
        x = torch.stack([torch.from_numpy((self.data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        return x.to(device), y.to(device)

print("Initializing Data Loaders...", flush=True)
train_loader = MemMapLoader('train')
val_loader = MemMapLoader('val')

# --- 3. MODEL INIT ---
print(f"Initializing XXL Model: {args.n_layer}L, {args.n_embd}D, {n_head}H...", flush=True)
config = GPTConfig(
    vocab_size=vocab_size, 
    block_size=1024, 
    n_layer=args.n_layer, 
    n_head=n_head, 
    n_embd=args.n_embd
)
model = GPT(config)
model.to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
scaler = torch.cuda.amp.GradScaler() 

best_val_loss = float('inf')
start_time = time.time()
train_log = []

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(50) 
        for k in range(50):
            X, Y = loader.get_batch(args.batch_size, 1024)
            with torch.cuda.amp.autocast():
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def evaluate_and_save(step):
    global best_val_loss
    print(f"Running evaluation at step {step}...", flush=True)
    losses = estimate_loss()
    print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}", flush=True)
    
    if losses['val'] < best_val_loss:
        best_val_loss = losses['val']
        print(f"  >> New Best! Saving checkpoint...", flush=True)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': vars(config),
            'iter': step,
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    train_log.append({
        "iter": step, 
        "val_loss": losses['val'].item(), 
        "time": time.time() - start_time
    })
    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump(train_log, f)

# --- 4. TRAINING LOOP ---
print("Starting Training Loop...", flush=True)

for iter in range(args.max_iters + 1): 
    
    if iter % 250 == 0:
        evaluate_and_save(iter)
        
    if iter == args.max_iters:
        break

    # Training Step
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(args.grad_accum):
        X, Y = train_loader.get_batch(args.batch_size, 1024)
        
        with torch.cuda.amp.autocast():
            logits, loss = model(X, Y)
            loss = loss / args.grad_accum
        
        scaler.scale(loss).backward()
    
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    if iter % 10 == 0:
        print(f"step {iter} | loss {loss.item() * args.grad_accum:.4f}", flush=True)

print("Done.", flush=True)