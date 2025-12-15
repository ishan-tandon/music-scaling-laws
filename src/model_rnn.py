"""
RNN (LSTM) Baseline model for comparison.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class RNNConfig:
    def __init__(self, vocab_size, n_layer, n_embd, block_size, dropout=0.0):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout

class RNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding table
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # LSTM Layers
        # batch_first=True means input is (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=config.n_embd,
            hidden_size=config.n_embd,
            num_layers=config.n_layer,
            batch_first=True,
            dropout=config.dropout if config.n_layer > 1 else 0
        )
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        
        x = self.token_embedding(idx) # (B, T, n_embd)
        
        out, _ = self.lstm(x) # out: (B, T, n_embd)
        
        logits = self.lm_head(out) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss