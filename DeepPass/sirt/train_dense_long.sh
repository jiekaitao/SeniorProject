#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sirt_dense_long_%j.log
#SBATCH --job-name=dense_ln

# Dense 172M baseline — SAME compute budget as SIRT long
# 47K steps total (30K+12K+5K) at K=1 (no recursion)
# Fair comparison: same data, same tokens, same architecture minus recursion

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Dense 172M Long Baseline ==="
echo "Started: $(date)"

$PYTHON sirt/train.py \
    --stage 1 \
    --max_steps 47000 \
    --batch_size 8 \
    --seq_len 4096 \
    --lr 6e-4 \
    --warmup_steps 1000 \
    --grad_accum 8 \
    --log_interval 200 \
    --save_interval 10000 \
    --save_dir sirt/checkpoints_dense_long \
    --data_dir sirt/data

echo "=== Finished: $(date) ==="
