#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_reason_mis_%j.log
#SBATCH --job-name=reas_mis

# Reasoning Probe: Mistral 7B
# Tests "think twice" hypothesis: prompt dup vs layer dup
# Uses trick reasoning questions (car wash, bat+ball, etc.)
# Core [28,29) — the +3.50 winner

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Reasoning Probe: Mistral 7B ===" && echo "Started: $(date)"

envs/deeppass/bin/python sirt/reasoning_probe.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --name mistral_7b \
    --core_start 28 --core_end 29 \
    --max_k 3

echo "=== Finished: $(date) ==="
