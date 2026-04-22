#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sirt_long_%j.log
#SBATCH --job-name=sirt_lng

# SIRT-172M LONG training: ~5B tokens across 3 stages
# Stage 1: 30K steps × 262K tok = 7.9B effective (cycles through 6.1B data)
# Stage 2: 12K steps = 3.1B
# Stage 3: 5K steps = 1.3B
# Total: ~12.3B token-steps (with data recycling)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== SIRT-172M Long Training (5B+ tokens) ==="
echo "Data shards: $(ls sirt/data/shard_*.bin | wc -l)"
echo "Started: $(date)"

# Stage 1: 30K steps, K=1
echo ""
echo "========== STAGE 1: 30K steps, K=1 =========="
$PYTHON sirt/train.py \
    --stage 1 \
    --max_steps 30000 \
    --batch_size 8 \
    --seq_len 4096 \
    --lr 6e-4 \
    --warmup_steps 1000 \
    --grad_accum 8 \
    --log_interval 200 \
    --save_interval 5000 \
    --save_dir sirt/checkpoints_long \
    --data_dir sirt/data

# Stage 2: 12K steps, K=1-3
echo ""
echo "========== STAGE 2: 12K steps, K=1-3 =========="
$PYTHON sirt/train.py \
    --stage 2 \
    --max_steps 12000 \
    --batch_size 8 \
    --seq_len 4096 \
    --lr 2e-4 \
    --warmup_steps 500 \
    --grad_accum 8 \
    --log_interval 200 \
    --save_interval 3000 \
    --save_dir sirt/checkpoints_long \
    --data_dir sirt/data \
    --resume sirt/checkpoints_long/stage1_final.pt

# Stage 3: 5K steps, adaptive
echo ""
echo "========== STAGE 3: 5K steps, adaptive =========="
$PYTHON sirt/train.py \
    --stage 3 \
    --max_steps 5000 \
    --batch_size 4 \
    --seq_len 4096 \
    --lr 5e-5 \
    --warmup_steps 200 \
    --grad_accum 8 \
    --log_interval 200 \
    --save_interval 2000 \
    --save_dir sirt/checkpoints_long \
    --data_dir sirt/data \
    --resume sirt/checkpoints_long/stage2_final.pt

echo ""
echo "=== SIRT Long Training Complete ==="
echo "Finished: $(date)"
