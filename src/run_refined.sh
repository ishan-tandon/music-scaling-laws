#!/bin/bash
set -e

# Common settings
ITERS=3000
BLOCK=1024

echo "================================================================"
echo "      STARTING FULL SCALING EXPERIMENT (OPTIMIZED LRs)"
echo "================================================================"

# 1. TINY (Standard LR)
# Small models are stable, so they can handle high LR (6e-4)
echo "[1/5] Training TINY..."
python src/train.py \
    --model_name tiny \
    --n_layer 4 --n_head 4 --n_embd 128 \
    --max_iters $ITERS --block_size $BLOCK \
    --batch_size 32 --grad_accum 1 \
    --learning_rate 6e-4

# 2. SMALL (Standard LR)
echo "[2/5] Training SMALL..."
python src/train.py \
    --model_name small \
    --n_layer 6 --n_head 6 --n_embd 288 \
    --max_iters $ITERS --block_size $BLOCK \
    --batch_size 32 --grad_accum 1 \
    --learning_rate 6e-4

# 3. MEDIUM (Lower LR)
# As we get bigger, we lower LR to prevent instability
echo "[3/5] Training MEDIUM..."
python src/train.py \
    --model_name medium \
    --n_layer 8 --n_head 8 --n_embd 512 \
    --max_iters $ITERS --block_size $BLOCK \
    --batch_size 32 --grad_accum 1 \
    --learning_rate 3e-4

# 4. LARGE (Lower LR)
# Physical Batch 8 * Accum 4 = Effective 32
echo "[4/5] Training LARGE..."
python src/train.py \
    --model_name large \
    --n_layer 12 --n_head 12 --n_embd 768 \
    --max_iters $ITERS --block_size $BLOCK \
    --batch_size 8 --grad_accum 4 \
    --learning_rate 2.5e-4

# 5. XL (Lowest LR)
# Physical Batch 4 * Accum 8 = Effective 32
echo "[5/5] Training XL..."
python src/train.py \
    --model_name xl \
    --n_layer 12 --n_head 16 --n_embd 1024 \
    --max_iters $ITERS --block_size $BLOCK \
    --batch_size 4 --grad_accum 8 \
    --learning_rate 1.5e-4

echo "================================================================"
echo "                 FULL EXPERIMENT COMPLETE"
echo "================================================================"