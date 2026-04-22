#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_trm_mis_hard_%j.log
#SBATCH --job-name=trm_mh

# Track C: TRM Mistral with harder training data
# GPT-5.4 Pro: use the same hard mix that improved PSRT-v2
# Best TRM config: [28,29), 500 steps
# New: train on 50% general + 25% math + 25% science instead of pure fineweb

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
export CUDA_VISIBLE_DEVICES=0
echo "=== TRM Mistral Hard Mix ===" && echo "Started: $(date)"

envs/deeppass/bin/python sirt/llm_to_trm.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --name mistral_7b_hardmix \
    --core_start 28 --core_end 29 \
    --max_steps 500 --lr 1e-4

echo "=== Finished: $(date) ==="
