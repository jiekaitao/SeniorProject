#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ft_llama3_v2_%j.log
#SBATCH --job-name=ft_ll3v2

# LLaMA 3 8B v2: much gentler — lr=1e-6, 500 steps
# Narrower core [10,13] (3 layers instead of 6)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== LLaMA 3 8B Recursion FT v2 ===" && echo "Started: $(date)"
envs/deeppass/bin/python sirt/recursion_finetune.py \
    --model /data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf \
    --name llama3_8b_v2 \
    --core_start 10 --core_end 13 \
    --max_steps 500 --lr 1e-6 --batch_size 1 --seq_len 1024
echo "=== Finished: $(date) ==="
