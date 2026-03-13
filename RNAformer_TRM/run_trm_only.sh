#!/bin/bash
# TRM-only training with reduced batch to fit in GPU memory
# Baseline uses 16.6GB at batch=2; TRM needs ~2x activation memory
# for dual-state (H=2, L=3) gradient-carrying passes.
# batch=1, accumulate=16 → effective batch=16 (same as baseline)
set -eo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/RNAformer:$PYTHONPATH"

echo "=========================================="
echo "  TRM RNAformer (dual-state+CGAR+ACT+RR)"
echo "  batch=1, accumulate=16 → effective=16"
echo "=========================================="
python3 train_comparison.py --variant trm \
  --h-cycles 2 --l-cycles 3 \
  --model-dim 128 --num-head 4 --n-layers 4 \
  --cycling 6 \
  --batch-size 1 --accumulate-grad 16 \
  --lr 1e-3 --warmup-steps 500 \
  --max-epochs 50 --max-len 200 \
  --val-check-interval 0.5 \
  --wandb-project RNAformer-TRM-comparison

echo ""
echo "=========================================="
echo "  DONE - Check wandb for comparison:"
echo "  https://wandb.ai/jiekaitao-university-of-florida/RNAformer-TRM-comparison"
echo "=========================================="
