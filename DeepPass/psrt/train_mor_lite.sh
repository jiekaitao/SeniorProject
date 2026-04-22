#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_mor_lite_%j.log
#SBATCH --job-name=morl

# PSRT-MoR-lite: shared attention + expert FFN + sequence router
# Phase A: K=2, uniform routing, expert FFNs only
# Phase B: K={2,3}, soft routing, all params
# Phase C: K={1-4}, top-2 routing, halting

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== PSRT-MoR-lite Training ===" && echo "Started: $(date)"

envs/deeppass/bin/python psrt/train_mor_lite.py \
    --size 172m \
    --total_steps 12000 \
    --batch_size 8 \
    --seq_len 1024 \
    --lr 3e-4

echo "=== Finished: $(date) ==="
