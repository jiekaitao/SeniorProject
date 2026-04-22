#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sirt_smoke_%j.log
#SBATCH --job-name=sirt_sm

# SIRT-170M GPU smoke test
# Runs 100 steps with synthetic data to verify:
# 1. Model fits on GPU
# 2. Loss decreases
# 3. β gating works
# 4. All 3 stages run without error

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== SIRT-170M Smoke Test ==="
echo "Started: $(date)"

# Stage 1: Fixed 1-recursion (50 steps)
echo ""
echo "=== Stage 1: Fixed 1-recursion ==="
$PYTHON sirt/train.py --stage 1 --synthetic --max_steps 50 --batch_size 4 --seq_len 512 \
    --log_interval 10 --save_interval 50 --save_dir sirt/checkpoints

# Stage 2: Curriculum 1-3 recursions (30 steps)
echo ""
echo "=== Stage 2: Recurrence curriculum ==="
$PYTHON sirt/train.py --stage 2 --synthetic --max_steps 30 --batch_size 4 --seq_len 512 \
    --log_interval 10 --save_interval 30 --save_dir sirt/checkpoints \
    --resume sirt/checkpoints/stage1_final.pt

# Stage 3: Adaptive halting (20 steps)
echo ""
echo "=== Stage 3: Adaptive halting ==="
$PYTHON sirt/train.py --stage 3 --synthetic --max_steps 20 --batch_size 2 --seq_len 512 \
    --log_interval 10 --save_interval 20 --save_dir sirt/checkpoints \
    --resume sirt/checkpoints/stage2_final.pt

echo ""
echo "=== SIRT Smoke Test Complete ==="
echo "Finished: $(date)"
