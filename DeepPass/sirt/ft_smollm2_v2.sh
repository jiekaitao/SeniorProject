#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ft_smollm2_v2_%j.log
#SBATCH --job-name=ft_sm2v2

# SmolLM2-360M v2: uses model's own tokenizer (fixes vocab mismatch)
# Gentler: lr=5e-6, 1000 steps, weighted toward K=1

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== SmolLM2-360M Recursion FT v2 ===" && echo "Started: $(date)"
envs/deeppass/bin/python sirt/recursion_finetune.py \
    --model HuggingFaceTB/SmolLM2-360M-Instruct \
    --name smollm2_360m_v2 \
    --core_start 10 --core_end 16 \
    --max_steps 1000 --lr 5e-6 --batch_size 2 --seq_len 1024
echo "=== Finished: $(date) ==="
