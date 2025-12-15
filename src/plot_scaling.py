import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

configs = {
    'tiny':   dict(n_layer=4,  n_head=4,  n_embd=128),
    'small':  dict(n_layer=6,  n_head=6,  n_embd=288),
    'medium': dict(n_layer=8,  n_head=8,  n_embd=512),
    'large':  dict(n_layer=12, n_head=12, n_embd=768),
    'xl':     dict(n_layer=12, n_head=16, n_embd=1024),
}

vocab_size = 99 
block_size = 1024

def get_params(conf):
    """Calculates parameter count without loading the massive model"""
    # Embedding
    n_params = vocab_size * conf['n_embd'] # wte
    n_params += block_size * conf['n_embd'] # wpe
    
    # Transformer Blocks
    # per block: ln1, attn, ln2, mlp
    # attn: c_attn (3*embd^2), c_proj (embd^2) -> 4*embd^2
    # mlp: c_fc (4*embd^2), c_proj (4*embd^2) -> 8*embd^2
    # layernorms: 2 * 2 * embd (weight+bias)
    params_per_block = 12 * conf['n_embd']**2 + 4 * conf['n_embd']
    n_params += conf['n_layer'] * params_per_block
    
    # Final LayerNorm
    n_params += 2 * conf['n_embd']
    
    return n_params

# Data Storage
results = []
training_curves = {}

print("Reading results...")
for name, conf in configs.items():
    path = f'checkpoints/{name}/stats.json'
    if not os.path.exists(path):
        print(f"Warning: {name} not found, skipping.")
        continue
        
    with open(path, 'r') as f:
        log = json.load(f)
        
    if not log:
        print(f"Warning: {name} log is empty.")
        continue

    final_val = log[-1]['val_loss']
    params = get_params(conf)
    
    results.append((params, final_val, name))
    training_curves[name] = log
    
    print(f"Model: {name:8s} | Params: {params/1e6:6.2f}M | Val Loss: {final_val:.4f}")

if not results:
    print("No results found! Check your checkpoints folder.")
    exit()

results.sort(key=lambda x: x[0])
N = np.array([x[0] for x in results])
L = np.array([x[1] for x in results])
names = [x[2] for x in results]

# --- 1. SCALING LAW FIT ---
def power_law(n, a, alpha, c):
    return a * (n ** -alpha) + c

# Initial guess for curve fitting
try:
    # Try to fit the curve
    popt, _ = curve_fit(power_law, N, L, p0=[10, 0.1, 0.5], maxfev=10000)
    a_fit, alpha_fit, c_fit = popt
    fitted = True
    print(f"\nSCALING LAW FOUND: alpha = {alpha_fit:.4f}")
except Exception as e:
    print(f"\nCurve fit failed: {e}")
    fitted = False
    alpha_fit = 0 # Default for safety

# --- PLOTTING ---
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Scaling Law
axs[0].scatter(N, L, color='red', s=100, zorder=5)
for i, txt in enumerate(names):
    axs[0].annotate(txt, (N[i], L[i]), xytext=(0, 10), textcoords='offset points', ha='center')

if fitted:
    x_range = np.logspace(np.log10(min(N)*0.8), np.log10(max(N)*1.2), 100)
    y_range = power_law(x_range, *popt)
    axs[0].plot(x_range, y_range, 'b--', label=f'Fit: $\\alpha={alpha_fit:.3f}$')
    axs[0].set_title(f'Scaling Law ($\\alpha={alpha_fit:.3f}$)')
else:
    axs[0].set_title('Scaling Law (No Fit)')

axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel('Parameters (N)')
axs[0].set_ylabel('Validation Loss (L)')
axs[0].grid(True, which="both", alpha=0.3)
axs[0].legend()

# Plot 2: Training Curves
for name, log in training_curves.items():
    steps = [entry['iter'] for entry in log]
    val_loss = [entry['val_loss'] for entry in log]
    axs[1].plot(steps, val_loss, label=name)

axs[1].set_title('Training Dynamics')
axs[1].set_xlabel('Steps')
axs[1].set_ylabel('Validation Loss')
axs[1].set_yscale('log')
axs[1].grid(True, alpha=0.3)
axs[1].legend()

for name, log in training_curves.items():
    if log and 'gpu_mem_gb' in log[0]:
        steps = [entry['iter'] for entry in log]
        mem = [entry['gpu_mem_gb'] for entry in log]
        axs[2].plot(steps, mem, label=name)

axs[2].set_title('GPU Memory Footprint')
axs[2].set_xlabel('Steps')
axs[2].set_ylabel('VRAM Usage (GB)')
axs[2].grid(True, alpha=0.3)
axs[2].legend()

plt.tight_layout()
plt.savefig('scaling_results.png')
print("Plot saved to scaling_results.png")