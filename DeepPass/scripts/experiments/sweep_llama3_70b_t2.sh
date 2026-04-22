#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_llama3_70b_t2_%j.log
#SBATCH --job-name=ll3_t2
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== LLaMA 3 70B Tier 2 (full neuron analysis) ===" && echo "Started: $(date)"
# Uses pre-downloaded model, skip SBUID (already done), just Tier 2
envs/deeppass/bin/python scripts/experiments/universal_sweep.py \
    --model /data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-70B-Instruct-hf \
    --name llama3_70b --max_blocks 10
echo "=== Finished: $(date) ==="
