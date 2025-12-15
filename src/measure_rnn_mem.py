import torch
import os
import json
from model_rnn import RNNConfig, RNN

configs = {
    'rnn_tiny':   dict(n_layer=2, n_embd=256),
    'rnn_small':  dict(n_layer=3, n_embd=512),
    'rnn_medium': dict(n_layer=4, n_embd=896),
    'rnn_large':  dict(n_layer=5, n_embd=1472),
}

block_size = 128
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"{'Model':<12} | {'VRAM (GB)':<10}")
print("-" * 25)

for name, conf in configs.items():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with open('data/vocab.json', 'r') as f:
        vocab_size = len(json.load(f))

    config = RNNConfig(vocab_size=vocab_size, block_size=block_size, **conf)
    model = RNN(config)
    model.to(device)
    model.train()

    x = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)
    y = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)

    logits, loss = model(x, y)
    loss.backward()

    mem = 0
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"{name:<12} | {mem:<10.2f}")

    del model, logits, loss