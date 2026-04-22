#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ft_smollm2_%j.log
#SBATCH --job-name=ft_sm2

# SmolLM2-360M recursion fine-tune
# 32 layers, core=[10,16] (mid layers, 6 layers)
# Best block from sweep was (26,27) but for recursion we want a wider core

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== SmolLM2-360M Recursion Fine-Tune ===" && echo "Started: $(date)"
envs/deeppass/bin/python sirt/recursion_finetune.py \
    --model HuggingFaceTB/SmolLM2-360M-Instruct \
    --name smollm2_360m \
    --core_start 10 --core_end 16 \
    --max_steps 2000 --lr 1e-5 --batch_size 2 --seq_len 2048
echo "=== Finished: $(date) ==="
