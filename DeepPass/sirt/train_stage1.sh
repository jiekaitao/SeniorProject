#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sirt_stage1_%j.log
#SBATCH --job-name=sirt_s1

# SIRT-172M Stage 1: Fixed 1-recursion LM training
# Target: ~2B tokens (about 12-16h on 1 B200)
# Effective batch: 4 * 4096 * 8 = 131K tokens/step
# Steps: 2B / 131K ≈ 15,000 steps

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== SIRT-172M Stage 1 Training ==="
echo "Started: $(date)"

# Check data exists
if [ ! -f sirt/data/shard_0000.bin ]; then
    echo "ERROR: No data shards found. Run prepare_data.sh first."
    exit 1
fi

$PYTHON sirt/train.py \
    --stage 1 \
    --max_steps 15000 \
    --batch_size 4 \
    --seq_len 4096 \
    --lr 3e-4 \
    --warmup_steps 500 \
    --grad_accum 8 \
    --log_interval 50 \
    --save_interval 2500 \
    --save_dir sirt/checkpoints \
    --data_dir sirt/data

echo "=== Finished: $(date) ==="
