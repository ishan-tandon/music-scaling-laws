import torch
import gc
from model import GPTConfig, GPT

candidates = [
    # Baseline: Your current XL
    {"name": "XL (Baseline)", "n_layer": 12, "n_embd": 1024}, 
    
    # Candidate 1: XXL (~230M Params)
    {"name": "XXL-200M",      "n_layer": 16, "n_embd": 1280},
    
    # Candidate 2: XXXL (~350M Params)
    {"name": "XXL-350M",      "n_layer": 24, "n_embd": 1280},
    
    # Candidate 3: Monster (~500M Params)
    {"name": "XXL-500M",      "n_layer": 24, "n_embd": 1600},
]

# Physical batch size of 1 is the ultimate efficiency hack.
# We will use Gradient Accumulation to make up the difference later.
PHYSICAL_BATCH = 1 
BLOCK_SIZE = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_params(model):
    return sum(p.numel() for p in model.parameters())

def attempt_config(conf):
    print(f"\n--- Testing {conf['name']} ---")
    try:
        # 1. Clean previous memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # 2. Init Model
        config = GPTConfig(
            vocab_size=128, # Dummy vocab
            block_size=BLOCK_SIZE,
            n_layer=conf['n_layer'],
            n_head=conf['n_embd'] // 64, # Auto-calc heads (64 dim per head)
            n_embd=conf['n_embd']
        )
        model = GPT(config)
        model.to(DEVICE)
        
        params = get_params(model)
        print(f"Parameters: {params/1e6:.2f}M")
        
        # 3. Dummy Forward/Backward Pass
        scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        x = torch.randint(0, 100, (PHYSICAL_BATCH, BLOCK_SIZE)).to(DEVICE)
        y = torch.randint(0, 100, (PHYSICAL_BATCH, BLOCK_SIZE)).to(DEVICE)
        
        with torch.cuda.amp.autocast():
            _, loss = model(x, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 4. Measure Memory
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"SUCCESS! Peak Memory: {mem:.2f} GB")
        return True, params
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("FAILED: CUDA Out of Memory")
            return False, 0
        else:
            print(f"FAILED: Other error ({e})")
            return False, 0

print(f"Starting Stress Test on {torch.cuda.get_device_name(0)}...")
print(f"Physical Batch: {PHYSICAL_BATCH} (Mixed Precision ON)")

valid_models = []
for conf in candidates:
    success, params = attempt_config(conf)
    if success:
        valid_models.append((conf, params))
    else:
        break

if valid_models:
    best_conf, best_params = valid_models[-1]
    print("\n" + "="*40)
    print(f"WINNER: {best_conf['name']} ({best_params/1e6:.1f}M Params)")
    print("Configuration details:")
    print(f"n_layer: {best_conf['n_layer']}")
    print(f"n_embd:  {best_conf['n_embd']}")
    print(f"n_head:  {best_conf['n_embd'] // 64}")
    print("="*40)
else:
    print("Even the baseline failed? Check your GPU.")

"""
rohan@jsba:~/music-scaling-laws$ docker compose exec dev python src/find_xxl.py
WARN[0000] /home/rohan/music-scaling-laws/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
Starting Stress Test on NVIDIA GeForce RTX 3090...
Physical Batch: 1 (Mixed Precision ON)

--- Testing XL (Baseline) ---
number of parameters: 151.29M
Parameters: 152.34M
SUCCESS! Peak Memory: 3.06 GB

--- Testing XXL-200M ---
number of parameters: 315.01M
Parameters: 316.32M
SUCCESS! Peak Memory: 6.48 GB

--- Testing XXL-350M ---
number of parameters: 472.42M
Parameters: 473.74M
SUCCESS! Peak Memory: 9.72 GB

--- Testing XXL-500M ---
number of parameters: 737.99M
Parameters: 739.63M
SUCCESS! Peak Memory: 15.10 GB

========================================
WINNER: XXL-500M (739.6M Params)
Configuration details:
n_layer: 24
n_embd:  1600
n_head:  25
========================================
"""