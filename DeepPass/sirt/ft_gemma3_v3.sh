#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ft_gemma3_v3_%j.log
#SBATCH --job-name=ft_g3v3

# Gemma3-27B v3: fixed token_type_ids in training loop

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Gemma3-27B Recursion FT v3 ===" && echo "Started: $(date)"
envs/deeppass/bin/python sirt/recursion_finetune.py \
    --model models/full/gemma-3-27b-it \
    --name gemma3_27b_v3 \
    --core_start 11 --core_end 14 \
    --max_steps 300 --lr 1e-6 --batch_size 1 --seq_len 512
echo "=== Finished: $(date) ==="
