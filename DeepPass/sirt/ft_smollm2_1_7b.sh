#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ft_smollm2_1_7b_%j.log
#SBATCH --job-name=ft_s17b

# SmolLM2-1.7B — the largest SmolLM, SOTA sub-2B
# Much stronger baseline than 360M, recursion might show clearer benefit

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== SmolLM2-1.7B Recursion FT ===" && echo "Started: $(date)"
envs/deeppass/bin/python sirt/recursion_finetune.py \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --name smollm2_1_7b \
    --core_start 8 --core_end 14 \
    --max_steps 500 --lr 5e-7 --batch_size 1 --seq_len 1024
echo "=== Finished: $(date) ==="
