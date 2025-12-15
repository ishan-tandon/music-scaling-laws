import matplotlib.pyplot as plt
import numpy as np
import json
import os
from collections import Counter

# --- CONFIG ---
DATA_PATH = os.path.join('data', 'train.bin')
VOCAB_PATH = os.path.join('data', 'vocab.json')
OUTPUT_IMG = "dataset_distribution.png"
SAMPLE_SIZE = 10_000_000

def plot_token_distribution():
    print(f"Loading vocab from {VOCAB_PATH}...")
    if not os.path.exists(VOCAB_PATH):
        print("Error: vocab.json not found.")
        return

    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
    
    itos = {}
    for i, ch in enumerate(vocab):
        if ch == '\n':
            itos[i] = '\\n'
        elif ch == ' ':
            itos[i] = 'Space'
        else:
            itos[i] = ch

    print(f"Reading first {SAMPLE_SIZE} tokens from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print("Error: train.bin not found.")
        return

    data = np.memmap(DATA_PATH, dtype=np.uint16, mode='r')
    
    limit = min(len(data), SAMPLE_SIZE)
    sample = data[:limit]

    print("Counting tokens...")
    counts = Counter(sample)

    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    labels = [itos.get(idx, str(idx)) for idx, count in sorted_items]
    values = [count for idx, count in sorted_items]

    top_n = 40
    labels = labels[:top_n]
    values = values[:top_n]

    print(f"Generating graph for top {top_n} tokens...")
    plt.figure(figsize=(12, 6))
    
    plt.bar(range(len(values)), values, color='skyblue', edgecolor='navy')
    
    plt.xticks(range(len(labels)), labels, rotation=0, fontsize=9, fontname='monospace')
    
    plt.title(f'Token Distribution in ABC Dataset (Top {top_n} Tokens)', fontsize=14)
    plt.xlabel('Token (Character)', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMG)
    print(f"Graph saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    plot_token_distribution()