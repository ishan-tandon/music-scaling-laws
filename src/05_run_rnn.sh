#!/bin/bash
set -e

# --- EQUIVALENCE MATH ---
# Transformer: 3000 steps * 32 batch * 1024 ctx = 98,304,000 tokens
# RNN:         3000 steps * 256 batch * 128 ctx = 98,304,000 tokens
# We use Batch 256 (Physical 32 * Accum 8) to match the data volume exactly.

ITERS=3000
BLOCK=128     # Standard LSTM context
BATCH=32      # Physical batch
ACCUM=8       # 32 * 8 = 256 Effective Batch
LR=1e-3       # LSTMs usually prefer higher LR than Transformers

echo "================================================================"
echo "           STARTING RNN SCALING STUDY (100M TOKENS)"
echo "================================================================"

# 1. RNN TINY (Match ~1M Params)
# L=2, Embd=256 -> ~1M params
echo "[1/4] Training RNN TINY..."
python src/train_rnn.py \
    --model_name rnn_tiny \
    --n_layer 2 --n_embd 256 \
    --max_iters $ITERS --block_size $BLOCK \
    --batch_size $BATCH --grad_accum $ACCUM \
    --learning_rate $LR

# 2. RNN SMALL (Match ~6M Params)
# L=3, Embd=512 -> ~6.5M params
echo "[2/4] Training RNN SMALL..."
python src/train_rnn.py \
    --model_name rnn_small \
    --n_layer 3 --n_embd 512 \
    --max_iters $ITERS --block_size $BLOCK \
    --batch_size $BATCH --grad_accum $ACCUM \
    --learning_rate $LR

# 3. RNN MEDIUM (Match ~25M Params)
# L=4, Embd=896 -> ~26M params
echo "[3/4] Training RNN MEDIUM..."
python src/train_rnn.py \
    --model_name rnn_medium \
    --n_layer 4 --n_embd 896 \
    --max_iters $ITERS --block_size $BLOCK \
    --batch_size $BATCH --grad_accum $ACCUM \
    --learning_rate $LR

# 4. RNN LARGE (Match ~85M Params)
# L=5, Embd=1472 -> ~87M params
# Note: Lower LR slightly for stability on large LSTM
echo "[4/4] Training RNN LARGE..."
python src/train_rnn.py \
    --model_name rnn_large \
    --n_layer 5 --n_embd 1472 \
    --max_iters $ITERS --block_size $BLOCK \
    --batch_size $BATCH --grad_accum $ACCUM \
    --learning_rate 5e-4

echo "================================================================"
echo "                  RNN EXPERIMENTS COMPLETE"
echo "================================================================"