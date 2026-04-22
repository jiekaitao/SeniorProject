#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_psrt_reason_%j.log
#SBATCH --job-name=psrt_r

# PSRT Reasoning: Train on maze/arithmetic/logic/counting/pattern
# 70% reasoning tasks + 30% general text
# Key question: does the model learn higher K for harder problems?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== PSRT Reasoning Training ===" && echo "Started: $(date)"

envs/deeppass/bin/python psrt/train_reasoning.py \
    --size 172m \
    --total_steps 15000 \
    --batch_size 8 \
    --seq_len 512 \
    --lr 3e-4 \
    --halt_penalty 0.0005

echo "=== Finished: $(date) ==="
