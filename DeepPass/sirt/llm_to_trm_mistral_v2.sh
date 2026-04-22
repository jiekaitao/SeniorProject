#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_trm_mis_v2_%j.log
#SBATCH --job-name=trm_m2

# LLM-to-TRM v2: Mistral 7B — more training steps
# v1 got +0.62 (K=2) and +0.68 (K=3) with only 500 steps
# v2: 1500 steps, lower LR for longer training

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
export CUDA_VISIBLE_DEVICES=0
echo "=== LLM-to-TRM v2: Mistral 7B (1500 steps) ===" && echo "Started: $(date)"

envs/deeppass/bin/python sirt/llm_to_trm.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --name mistral_7b_v2 \
    --core_start 28 --core_end 29 \
    --max_steps 1500 --lr 5e-5

echo "=== Finished: $(date) ==="
