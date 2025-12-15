import torch
import numpy as np
import os
import math
import json
from model import GPT, GPTConfig

# --- CONFIG ---
# POINTING TO THE NEW TRUE TEST SET
DATA_PATH = os.path.join('data', 'test.bin') 
BATCH_SIZE = 32
BLOCK_SIZE = 1024
EVAL_ITERS = 200 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- SETUP ---
print(f"--- PERPLEXITY EVALUATION ON {DEVICE.upper()} ---")

# 1. Load Vocab
with open('data/vocab.json', 'r') as f:
    vocab = json.load(f)

# 2. Load Data
if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} not found. Did you run prepare_test.py?")
    exit()
    
# We use memmap to handle large files without loading everything into RAM
data = np.memmap(DATA_PATH, dtype=np.uint16, mode='r')
print(f"Loaded TEST data from {DATA_PATH} ({len(data)} tokens)")

# 3. Load Model
print("Loading model checkpoint...")
checkpoint = torch.load('checkpoints/xxl_final/ckpt.pt', map_location=DEVICE)

config = GPTConfig(
    block_size=1024, 
    vocab_size=len(vocab), 
    n_layer=24, 
    n_head=25, 
    n_embd=1600
)
model = GPT(config)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# --- CALCULATION LOOP ---
def get_batch():
    # Randomly sample chunks from the test set
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_perplexity():
    print(f"Calculating loss over {EVAL_ITERS} batches...")
    losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
        X, Y = get_batch()
        logits, loss = model(X, Y)
        losses[k] = loss.item()
        
        if (k+1) % 50 == 0:
            print(f"Processed {k+1}/{EVAL_ITERS} batches...")
    
    mean_loss = losses.mean()
    perplexity = math.exp(mean_loss)
    return mean_loss, perplexity

# --- EXECUTE ---
loss, perp = estimate_perplexity()

print("\n" + "="*40)
print("FINAL TRUE TEST SET METRICS")
print("="*40)
print(f"Final Loss:       {loss:.4f}")
print(f"Final Perplexity: {perp:.4f}")
print("="*40)

"""
rohan@jsba:~/music-scaling-laws$ docker compose exec dev python src/calc_perplexity.py
WARN[0000] /home/rohan/music-scaling-laws/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
--- PERPLEXITY EVALUATION ON CUDA ---
Loaded TEST data from data/test.bin (38821936 tokens)
Loading model checkpoint...
number of parameters: 737.94M
Calculating loss over 200 batches...
Processed 50/200 batches...
Processed 100/200 batches...
Processed 150/200 batches...
Processed 200/200 batches...

========================================
FINAL TRUE TEST SET METRICS
========================================
Final Loss:       0.3489
Final Perplexity: 1.4174
========================================
"""