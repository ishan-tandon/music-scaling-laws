import matplotlib.pyplot as plt
import re
import sys
import os

# --- CONFIG ---
DEFAULT_LOG_FILE = "src/fulloutput.txt"
OUTPUT_IMG = "training_curve.png"

def parse_log(filename):
    eval_steps = []
    eval_train_loss = []
    eval_val_loss = []
    
    running_steps = []
    running_loss = []

    # Regex for "step 250: train 0.9057, val 1.6165"
    pattern_eval = re.compile(r"step\s+(\d+):\s+train\s+([\d\.]+),\s+val\s+([\d\.]+)")
    
    # Regex for "step 10 | loss 4.3443"
    pattern_running = re.compile(r"step\s+(\d+)\s+\|\s+loss\s+([\d\.]+)")

    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return None, None, None, None, None

    print(f"Reading {filename}...")
    with open(filename, 'r') as f:
        for line in f:
            # Check for Evaluation Line
            match_eval = pattern_eval.search(line)
            if match_eval:
                eval_steps.append(int(match_eval.group(1)))
                eval_train_loss.append(float(match_eval.group(2)))
                eval_val_loss.append(float(match_eval.group(3)))
                continue

            # Check for Running Line
            match_running = pattern_running.search(line)
            if match_running:
                running_steps.append(int(match_running.group(1)))
                running_loss.append(float(match_running.group(2)))

    return eval_steps, eval_train_loss, eval_val_loss, running_steps, running_loss

def plot_graph(eval_steps, eval_train, eval_val, run_steps, run_loss):
    if not eval_steps:
        print("No evaluation data found! Make sure your log file has lines like 'step 250: train X, val Y'")
        return

    plt.figure(figsize=(12, 7))
    
    plt.plot(run_steps, run_loss, label='Running Train Loss', color='gray', alpha=0.3, linewidth=0.5)
    
    plt.plot(eval_steps, eval_train, label='Eval Train Loss', color='blue', marker='o', linewidth=2)
    
    plt.plot(eval_steps, eval_val, label='Validation Loss', color='red', marker='o', linewidth=2)
    
    plt.title('Training vs Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (Cross Entropy)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    min_val = min(eval_val)
    min_idx = eval_val.index(min_val)
    best_step = eval_steps[min_idx]
    plt.annotate(f'Best Val: {min_val:.4f}', 
                 xy=(best_step, min_val), 
                 xytext=(best_step, min_val + 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.savefig(OUTPUT_IMG)
    print(f"Success! Graph saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG_FILE
    
    data = parse_log(log_file)
    if data[0]: 
        plot_graph(*data)