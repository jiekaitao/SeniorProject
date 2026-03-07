#!/bin/bash
# Sequential training: baseline then TRM, both to same wandb project
# Using bf16-mixed to avoid NaN at deeper cycling depths (fp16 overflows at ±65504)
set -eo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/mnt/Data/GitHub/SeniorProject:/mnt/Data/GitHub/SeniorProject/RNAformer:$PYTHONPATH"

# Shared args (except batch size - TRM needs smaller batch for dual-state memory)
BASE_ARGS="--model-dim 128 --num-head 4 --n-layers 4 \
  --cycling 6 \
  --lr 1e-3 --warmup-steps 500 \
  --max-epochs 50 --max-len 200 \
  --val-check-interval 0.5 \
  --wandb-project RNAformer-TRM-comparison"

echo "=========================================="
echo "  Phase 1: BASELINE RNAformer (cycling=6)"
echo "  batch=2, accumulate=8 → effective=16"
echo "=========================================="
python3 train_comparison.py --variant baseline \
  --batch-size 2 --accumulate-grad 8 \
  $BASE_ARGS

echo ""
echo "=========================================="
echo "  Phase 2: TRM RNAformer (dual-state+CGAR+ACT+RR)"
echo "  batch=1, accumulate=16 → effective=16"
echo "=========================================="
python3 train_comparison.py --variant trm \
  --h-cycles 2 --l-cycles 3 \
  --batch-size 1 --accumulate-grad 16 \
  $BASE_ARGS

echo ""
echo "=========================================="
echo "  DONE - Check wandb for comparison:"
echo "  https://wandb.ai/jiekaitao-university-of-florida/RNAformer-TRM-comparison"
echo "=========================================="
