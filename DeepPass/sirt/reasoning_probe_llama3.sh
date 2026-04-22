#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_reason_ll3_%j.log
#SBATCH --job-name=reas_ll3

# Reasoning Probe: LLaMA 3 8B
# Tests "think twice" on trick questions (car wash, bat+ball, etc.)
# Core [10,13) — same as other LLaMA experiments
# Compares: K=1 baseline vs prompt duplication vs layer duplication K=2,3

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Reasoning Probe: LLaMA 3 8B ===" && echo "Started: $(date)"

envs/deeppass/bin/python sirt/reasoning_probe.py \
    --model /data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf \
    --name llama3_8b \
    --core_start 10 --core_end 13 \
    --max_k 3

echo "=== Finished: $(date) ==="
