#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sirt_full_%j.log
#SBATCH --job-name=sirt_ful

# SIRT-172M Full Training Run
# 3 stages on ~5B tokens total
# Targets competitive with SmolLM-135M class

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== SIRT-172M Full Training ==="
echo "Started: $(date)"
echo "Data shards: $(ls sirt/data/shard_*.bin | wc -l)"

# Stage 1: Fixed K=1 (60% of budget = 14K steps)
echo ""
echo "========== STAGE 1: Fixed K=1 =========="
$PYTHON sirt/train.py \
    --stage 1 \
    --max_steps 14000 \
    --batch_size 8 \
    --seq_len 4096 \
    --lr 6e-4 \
    --warmup_steps 500 \
    --grad_accum 8 \
    --log_interval 100 \
    --save_interval 3000 \
    --save_dir sirt/checkpoints_full \
    --data_dir sirt/data

# Stage 2: Curriculum K=1-3 (25% = 6K steps)
echo ""
echo "========== STAGE 2: Recurrence Curriculum =========="
$PYTHON sirt/train.py \
    --stage 2 \
    --max_steps 6000 \
    --batch_size 8 \
    --seq_len 4096 \
    --lr 2e-4 \
    --warmup_steps 200 \
    --grad_accum 8 \
    --log_interval 100 \
    --save_interval 2000 \
    --save_dir sirt/checkpoints_full \
    --data_dir sirt/data \
    --resume sirt/checkpoints_full/stage1_final.pt

# Stage 3: Adaptive halting (15% = 3K steps)
echo ""
echo "========== STAGE 3: Adaptive Halting =========="
$PYTHON sirt/train.py \
    --stage 3 \
    --max_steps 3000 \
    --batch_size 4 \
    --seq_len 4096 \
    --lr 5e-5 \
    --warmup_steps 100 \
    --grad_accum 8 \
    --log_interval 100 \
    --save_interval 1000 \
    --save_dir sirt/checkpoints_full \
    --data_dir sirt/data \
    --resume sirt/checkpoints_full/stage2_final.pt

echo ""
echo "=== SIRT Full Training Complete ==="
echo "Finished: $(date)"
