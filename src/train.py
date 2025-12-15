"""
Training script for the music scaling laws project.
Includes:
- Argument parsing for model size and learning rate
- Gradient Accumulation for large models
- Memory logging
- Final evaluation block
"""
import os
import time
import math
import pickle
import json
import argparse
import numpy as np
import torch

from model import GPTConfig, GPT

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True, help='Name for this run (e.g. "small")')
parser.add_argument('--n_layer', type=int, required=True)
parser.add_argument('--n_head', type=int, required=True)
parser.add_argument('--n_embd', type=int, required=True)
parser.add_argument('--max_iters', type=int, default=3000, help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--block_size', type=int, default=1024)
parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--learning_rate', type=float, default=6e-4, help='Base learning rate')
args = parser.parse_args()

# --- Configuration ---
out_dir = f'checkpoints/{args.model_name}'
eval_interval = 250
log_interval = 10 
eval_iters = 200

# Use the argument for LR, and scale min_lr accordingly
learning_rate = args.learning_rate 
min_lr = learning_rate / 10 

warmup_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compile = False 

os.makedirs(out_dir, exist_ok=True)

# --- Data Loading ---
data_dir = 'data'
with open(os.path.join(data_dir, 'vocab.json'), 'r') as f:
    vocab = json.load(f)
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }

def encode(s):
    return [stoi[c] for c in s]

class TextLoader:
    def __init__(self, split):
        self.path = os.path.join(data_dir, f'{split}.txt')
        self.file_size = os.path.getsize(self.path)
        self.f = open(self.path, 'r', encoding='utf-8')
        
    def get_batch(self, batch_size, block_size):
        data_batch = []
        target_batch = []
        for _ in range(batch_size):
            idx = torch.randint(0, self.file_size - block_size - 1, (1,)).item()
            self.f.seek(idx)
            chunk = self.f.read(block_size + 1)
            while len(chunk) < block_size + 1:
                self.f.seek(0)
                chunk = self.f.read(block_size + 1)
            try:
                encoded = encode(chunk)
            except KeyError:
                continue
            x = torch.tensor(encoded[:-1], dtype=torch.long)
            y = torch.tensor(encoded[1:], dtype=torch.long)
            data_batch.append(x)
            target_batch.append(y)
        return torch.stack(data_batch), torch.stack(target_batch)

train_loader = TextLoader('train')
val_loader = TextLoader('val')

def get_batch(split):
    loader = train_loader if split == 'train' else val_loader
    x, y = loader.get_batch(args.batch_size, args.block_size)
    return x.to(device), y.to(device)

# --- Model ---
model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=args.block_size, bias=False, vocab_size=vocab_size, dropout=0.0)
config = GPTConfig(**model_args)
model = GPT(config)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > args.max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (args.max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# --- Training Loop ---
print(f"Starting training {args.model_name}...")
print(f"Config: Physical Batch={args.batch_size}, GradAccum={args.grad_accum}, LR={learning_rate}")

t0 = time.time()
best_val_loss = 1e9
train_log = []

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

optimizer.zero_grad(set_to_none=True)

# Helper for logging
def log_metrics(step):
    global best_val_loss
    losses = estimate_loss()
    mem_gb = 0
    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        
    print(f"\n[EVAL] step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, mem {mem_gb:.2f}GB\n")
    
    train_log.append({
        "iter": step,
        "train_loss": losses['train'].item(),
        "val_loss": losses['val'].item(),
        "lr": lr if 'lr' in locals() else min_lr,
        "time": time.time() - t0,
        "gpu_mem_gb": mem_gb
    })
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    if losses['val'] < best_val_loss:
        best_val_loss = losses['val']
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter': step,
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

# Main Loop
for iter in range(args.max_iters):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Periodic Eval
    if iter % eval_interval == 0:
        log_metrics(iter)

    # Gradient Accumulation
    for micro_step in range(args.grad_accum):
        X, Y = get_batch('train')
        logits, loss = model(X, Y)
        loss = loss / args.grad_accum
        loss.backward()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Print Progress
    if iter % log_interval == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        tokens_per_sec = (args.batch_size * args.grad_accum * args.block_size) / dt
        print(f"step {iter}/{args.max_iters} | loss {loss.item() * args.grad_accum:.4f} | {dt*1000:.1f}ms | {tokens_per_sec/1000:.1f}k tok/s")

# --- FINAL EVALUATION ---
print("Running final evaluation...")
log_metrics(args.max_iters)

# Save Stats
with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
    json.dump(train_log, f)
    
print(f"Training finished. Best Val Loss: {best_val_loss:.4f}")