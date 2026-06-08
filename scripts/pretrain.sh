#!/bin/bash
set -e

CONFIG=${1:-configs/350m.yaml}
BATCH_SIZE=${2:-8}
PRECISION=${3:-bf16}
SAVE_DIR=${4:-checkpoints}
DATA_MIX=${5:-configs/data_mix.yaml}

echo "=== Mamba-Attention Hybrid Training ==="
echo "Config:    $CONFIG"
echo "Batch:     $BATCH_SIZE"
echo "Precision: $PRECISION"
echo "Save dir:  $SAVE_DIR"
echo "Data mix:  $DATA_MIX"
echo "======================================"

python -m src.training.train \
    --config "$CONFIG" \
    --batch_size "$BATCH_SIZE" \
    --max_steps 1000000 \
    --warmup_steps 2000 \
    --lr 3e-4 \
    --min_lr 1e-5 \
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    --grad_accum 1 \
    --seed 42 \
    --data_mix "$DATA_MIX" \
    --precision "$PRECISION" \
    --log_interval 10 \
    --save_interval 1000 \
    --save_dir "$SAVE_DIR" \
    --wandb
