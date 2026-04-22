#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ft_gemma3_%j.log
#SBATCH --job-name=ft_g3

# Gemma3-27B recursion fine-tune
# 62 layers, core=[10,14] (mid layers, where L12 was the best helper)
# Already on disk at models/full/gemma-3-27b-it

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Gemma3-27B Recursion Fine-Tune ===" && echo "Started: $(date)"
envs/deeppass/bin/python sirt/recursion_finetune.py \
    --model models/full/gemma-3-27b-it \
    --name gemma3_27b \
    --core_start 10 --core_end 14 \
    --max_steps 500 --lr 2e-6 --batch_size 1 --seq_len 1024
echo "=== Finished: $(date) ==="
