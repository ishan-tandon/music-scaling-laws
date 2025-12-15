import torch
import json
import os
from torch.nn import functional as F
from model import GPT, GPTConfig

# --- CONFIG ---
NUM_SAMPLES = 100
MAX_TOKENS = 1024
TEMPERATURE = 1.0  
TOP_K = 200        
OUTPUT_DIR = "generated_samples"

# --- SETUP ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading Model on {device}...")

with open('data/vocab.json', 'r') as f: vocab = json.load(f)
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

checkpoint = torch.load('checkpoints/xxl_final/ckpt.pt', map_location=device)
config = GPTConfig(block_size=1024, vocab_size=len(vocab), n_layer=24, n_head=25, n_embd=1600)
model = GPT(config)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix): state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# --- GENERATION FUNCTION ---
def generate_sample(index):
    start_str = "X:"
    start_ids = [stoi[c] for c in start_str if c in stoi]
    x = torch.tensor([start_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(MAX_TOKENS):
            x_cond = x if x.size(1) <= config.block_size else x[:, -config.block_size:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / TEMPERATURE
            v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

    output_text = "".join([itos.get(int(i), '') for i in x[0].tolist()])
    filename = f"{OUTPUT_DIR}/sample_uncond_{index+1:03d}.abc"
    with open(filename, 'w') as f:
        f.write(output_text)
    return filename

# --- MAIN LOOP ---
print(f"\n--- GENERATING {NUM_SAMPLES} UNCONDITIONAL SAMPLES ---\n")
for i in range(NUM_SAMPLES):
    try:
        fname = generate_sample(i)
        print(f"[{i+1}/{NUM_SAMPLES}] Saved: {fname}")
    except Exception as e:
        print(f"[{i+1}/{NUM_SAMPLES}] Error: {e}")

print(f"\nDone! Check the '{OUTPUT_DIR}' folder.")