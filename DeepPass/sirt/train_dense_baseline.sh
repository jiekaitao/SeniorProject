#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sirt_dense_%j.log
#SBATCH --job-name=sirt_den

# Dense 172M baseline — same params, same data, no recursion
# 10 standard blocks (3 prelude + 3 core + 4 coda, all unique weights)
# Train Stage 1 only (K=1 fixed) for fair comparison

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Dense 172M Baseline Training ==="
echo "Started: $(date)"

$PYTHON sirt/train.py \
    --stage 1 \
    --max_steps 23000 \
    --batch_size 8 \
    --seq_len 4096 \
    --lr 6e-4 \
    --warmup_steps 500 \
    --grad_accum 8 \
    --log_interval 100 \
    --save_interval 5000 \
    --save_dir sirt/checkpoints_dense \
    --data_dir sirt/data

echo "=== Finished: $(date) ==="
