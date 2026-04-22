#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_stepdecay_ll3_%j.log
#SBATCH --job-name=decay_l

# Track A on LLaMA 3 8B: same step-decayed FFN sweep
# Core [10,13) — 3 layers
# Cross-model validation of the decay schedule finding

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Step-Decayed FFN: LLaMA 3 8B ===" && echo "Started: $(date)"

envs/deeppass/bin/python sirt/step_decayed_ffn.py \
    --model /data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf \
    --name llama3_8b \
    --core_start 10 --core_end 13

echo "=== Finished: $(date) ==="
