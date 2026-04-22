#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ft_mistral_%j.log
#SBATCH --job-name=ft_mis

# Mistral 7B recursion fine-tune
# THE model where raw dup already improves lm-eval
# Best block: (28,29). Core=[27,30] (3 layers around it)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Mistral 7B Recursion FT ===" && echo "Started: $(date)"
envs/deeppass/bin/python sirt/recursion_finetune.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --name mistral_7b \
    --core_start 27 --core_end 30 \
    --max_steps 500 --lr 1e-6 --batch_size 1 --seq_len 1024
echo "=== Finished: $(date) ==="
