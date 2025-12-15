import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("plasma", 4)

def plot_rnn_curves():
    rnn_dirs = sorted([d for d in os.listdir('checkpoints_rnn') if d.startswith('rnn_')])
    
    order = ['rnn_tiny', 'rnn_small', 'rnn_medium', 'rnn_large']
    rnn_dirs = [d for d in order if d in rnn_dirs]

    plt.figure(figsize=(10, 6))
    
    for name in rnn_dirs:
        path = f'checkpoints_rnn/{name}/stats.json'
        if not os.path.exists(path):
            print(f"Skipping {name}, no stats found.")
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        steps = [entry['iter'] for entry in data]
        val_loss = [entry['val_loss'] for entry in data]
        
        plt.plot(steps, val_loss, label=name.replace('rnn_', '').upper(), linewidth=2)

    plt.title('RNN Training Dynamics: Loss vs Steps', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Validation Loss', fontsize=14)
    plt.legend(title="Model Size", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    out_path = 'rnn_training_curves.png'
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    plot_rnn_curves()