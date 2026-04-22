#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_trm_mis_v3_%j.log
#SBATCH --job-name=trm_m3

# LLM-to-TRM v3: Mistral 7B — wider core [27,30) with TRM projections
# Previous: [28,29) raw dup was catastrophic but TRM projections might fix it
# The projections learn to split memory/reasoning, potentially taming the wider core

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
export CUDA_VISIBLE_DEVICES=0
echo "=== LLM-to-TRM v3: Mistral 7B (wider core [27,30)) ===" && echo "Started: $(date)"

envs/deeppass/bin/python sirt/llm_to_trm.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --name mistral_7b_wide \
    --core_start 27 --core_end 30 \
    --max_steps 1000 --lr 5e-5

echo "=== Finished: $(date) ==="
