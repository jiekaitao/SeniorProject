#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_stepdecay_lmeval_%j.log
#SBATCH --job-name=sd_eval

# lm-eval on Mistral with step-decayed FFN at K=2
# Schedule B: beta=[1.0, 0.25] (gave +2.01 on dual probe)
# Also runs baseline for comparison
# This is our "fastest publishable positive-control line"

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6

echo "=== Step-Decay lm-eval: Mistral 7B ===" && echo "Started: $(date)"

PYTHON=envs/deeppass/bin/python

# We need a custom script that applies step-decayed FFN during lm-eval
# For now, run the raw duplication lm-eval as a control
# The step-decay requires hooks that integrate with lm-eval's HFLM

# Run baseline first
$PYTHON scripts/experiments/lm_eval_runtime_dup.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --blocks "28,29" \
    --tasks leaderboard_bbh,leaderboard_math_hard,leaderboard_mmlu_pro,leaderboard_musr \
    --limit 0.0 \
    --output results/lm_eval/mistral_raw_dup_28_29.json

echo "=== Finished: $(date) ==="
