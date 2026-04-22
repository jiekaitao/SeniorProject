#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ft_llama3_8b_%j.log
#SBATCH --job-name=ft_ll3

# LLaMA 3 8B Instruct recursion fine-tune
# 32 layers, core=[8,14] (early-mid, where duplication helped most)
# Pre-downloaded on HiPerGator

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== LLaMA 3 8B Recursion Fine-Tune ===" && echo "Started: $(date)"
envs/deeppass/bin/python sirt/recursion_finetune.py \
    --model /data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf \
    --name llama3_8b \
    --core_start 8 --core_end 14 \
    --max_steps 1000 --lr 5e-6 --batch_size 1 --seq_len 2048
echo "=== Finished: $(date) ==="
