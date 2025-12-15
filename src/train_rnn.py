"""
Training script for the RNN (LSTM) baseline.
Features: Gradient Accumulation, Token Matching, Final Eval.
"""
import os
import time
import json
import argparse
import torch
from model_rnn import RNNConfig, RNN

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--n_embd', type=int, default=256)
parser.add_argument('--max_iters', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--block_size', type=int, default=128)
parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--learning_rate', type=float, default=1e-3)
args = parser.parse_args()

# --- Configuration ---
out_dir = f'checkpoints_rnn/{args.model_name}'
os.makedirs(out_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 250
log_interval = 10
eval_iters = 100

# --- Data Loading ---
data_dir = 'data'
with open(os.path.join(data_dir, 'vocab.json'), 'r') as f:
    vocab = json.load(f)
vocab_size = len(vocab)
stoi = { ch:i for i,ch in enumerate(vocab) }

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

# --- Model Init ---
config = RNNConfig(vocab_size=vocab_size, n_layer=args.n_layer, n_embd=args.n_embd, block_size=args.block_size)
model = RNN(config)
model.to(device)
print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

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

# --- Training Loop ---
print(f"Training RNN {args.model_name}...")
print(f"Eff. Batch: {args.batch_size * args.grad_accum} (Physical {args.batch_size} * Accum {args.grad_accum})")

t0 = time.time()
train_log = []

def log_metrics(step):
    losses = estimate_loss()
    print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    train_log.append({
        "iter": step, 
        "val_loss": losses['val'].item(),
        "time": time.time() - t0,
        # Log params once for the plotter
        "params": sum(p.numel() for p in model.parameters())
    })
    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump(train_log, f)

optimizer.zero_grad(set_to_none=True)

for iter in range(args.max_iters):
    
    # Periodic Eval
    if iter % eval_interval == 0:
        log_metrics(iter)

    # Gradient Accumulation Loop
    for micro_step in range(args.grad_accum):
        X, Y = get_batch('train')
        logits, loss = model(X, Y)
        loss = loss / args.grad_accum # Scale loss
        loss.backward()
    
    # Clip and Step
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if iter % log_interval == 0:
        dt = time.time() - t0
        print(f"step {iter} | loss {loss.item() * args.grad_accum:.4f}")

# --- FINAL EVALUATION ---
print("Running final evaluation...")
log_metrics(args.max_iters)
print("Done.")