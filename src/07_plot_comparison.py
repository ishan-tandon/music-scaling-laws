import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# --- 1. Load Transformer Data ---
trans_configs = {
    'tiny':   dict(n_layer=4,  n_head=4,  n_embd=128),
    'small':  dict(n_layer=6,  n_head=6,  n_embd=288),
    'medium': dict(n_layer=8,  n_head=8,  n_embd=512),
    'large':  dict(n_layer=12, n_head=12, n_embd=768),
    'xl':     dict(n_layer=12, n_head=16, n_embd=1024),
}

def get_trans_params(conf):
    # Simplified param calc matching previous script
    n_params = 99 * conf['n_embd'] + 1024 * conf['n_embd'] # embeddings
    params_per_block = 12 * conf['n_embd']**2 + 4 * conf['n_embd']
    n_params += conf['n_layer'] * params_per_block
    n_params += 2 * conf['n_embd']
    return n_params

trans_data = []
for name, conf in trans_configs.items():
    path = f'checkpoints/{name}/stats.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            log = json.load(f)
        if log:
            final_val = log[-1]['val_loss']
            params = get_trans_params(conf)
            trans_data.append((params, final_val, name))

trans_data.sort(key=lambda x: x[0])

# --- 2. Load RNN Data ---
rnn_data = []
rnn_dirs = [d for d in os.listdir('checkpoints_rnn') if d.startswith('rnn_')]

for name in rnn_dirs:
    path = f'checkpoints_rnn/{name}/stats.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            log = json.load(f)
        if log:
            final_val = log[-1]['val_loss']
            if 'params' in log[-1]:
                params = log[-1]['params']
                rnn_data.append((params, final_val, name))

rnn_data.sort(key=lambda x: x[0])

# --- 3. Power Law Fitting ---
def power_law(n, a, alpha, c):
    return a * (n ** -alpha) + c

def fit_and_plot(data, color, label, ax):
    N = np.array([x[0] for x in data])
    L = np.array([x[1] for x in data])
    
    # Plot points
    ax.scatter(N, L, color=color, s=100, zorder=5, label=f'{label} Data')
    
    # Try fit
    try:
        popt, _ = curve_fit(power_law, N, L, p0=[10, 0.1, 0.5], maxfev=10000)
        a, alpha, c = popt
        
        x_range = np.logspace(np.log10(min(N)*0.8), np.log10(max(N)*1.2), 100)
        y_range = power_law(x_range, *popt)
        
        ax.plot(x_range, y_range, color=color, linestyle='--', label=f'{label} Fit ($\\alpha={alpha:.3f}$)')
        print(f"{label} Alpha: {alpha:.4f}")
        return alpha
    except:
        print(f"Could not fit {label}")
        return 0

# --- 4. Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

# Plot Transformer
alpha_t = fit_and_plot(trans_data, 'blue', 'Transformer', ax)

# Plot RNN
alpha_r = fit_and_plot(rnn_data, 'red', 'LSTM (RNN)', ax)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Parameters (N)', fontsize=14)
ax.set_ylabel('Validation Loss (L)', fontsize=14)
ax.set_title(f'Scaling Laws: Transformer vs RNN\nTransformer $\\alpha={alpha_t:.3f}$ vs RNN $\\alpha={alpha_r:.3f}$', fontsize=16)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=12)

# Annotate points
for p, l, n in trans_data:
    ax.annotate(n, (p, l), xytext=(0, -15), textcoords='offset points', ha='center', color='blue', fontsize=8)
for p, l, n in rnn_data:
    ax.annotate(n, (p, l), xytext=(0, 10), textcoords='offset points', ha='center', color='red', fontsize=8)

plt.tight_layout()
plt.savefig('comparison_scaling.png')
print("\nPlot saved to comparison_scaling.png")

"""
rohan@jsba:~/music-scaling-laws$ docker compose exec dev python src/07_plot_comparison.py
WARN[0000] /home/rohan/music-scaling-laws/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
Transformer Alpha: 0.6270
LSTM (RNN) Alpha: 0.2016

Plot saved to comparison_scaling.png
"""