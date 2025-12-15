import os
import json
import numpy as np

# Define configs again to match names to params
configs = {
    'tiny':   dict(n_layer=4,  n_head=4,  n_embd=128),
    'small':  dict(n_layer=6,  n_head=6,  n_embd=288),
    'medium': dict(n_layer=8,  n_head=8,  n_embd=512),
    'large':  dict(n_layer=12, n_head=12, n_embd=768),
    'xl':     dict(n_layer=12, n_head=16, n_embd=1024),
}

print(f"{'Model':<12} | {'Params (M)':<10} | {'Speed (tok/s)':<15} | {'VRAM (GB)':<10}")
print("-" * 55)

# 1. Transformers
for name in configs:
    path = f'checkpoints/{name}/stats.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            log = json.load(f)
        
        # Calculate speed (avg time per iter over last 100 iters)
        # Batch=32, Block=1024 -> 32,768 tokens per iter
        times = [entry['time'] for entry in log[-100:] if 'time' in entry]
        avg_time = np.mean(times) if times else 0
        speed = 32768 / avg_time if avg_time > 0 else 0
        
        # Get Max Memory
        mem = max([entry.get('gpu_mem_gb', 0) for entry in log])
        
        # Get Params
        print(f"Trans-{name:<6} | {'?':<10} | {speed:<15.0f} | {mem:<10.2f}")

print("-" * 55)

# 2. RNNs
# RNNs: Batch=32*8=256, Block=128 -> 32,768 tokens per iter
rnn_dirs = sorted([d for d in os.listdir('checkpoints_rnn') if d.startswith('rnn_')])
for name in rnn_dirs:
    path = f'checkpoints_rnn/{name}/stats.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            log = json.load(f)
            
        times = [entry['time'] for entry in log[-100:] if 'time' in entry]
        avg_time = np.mean(times) if times else 0
        speed = 32768 / avg_time if avg_time > 0 else 0
        
        mem = 0
        
        print(f"{name:<12} | {'?':<10} | {speed:<15.0f} | {mem:<10.2f}")