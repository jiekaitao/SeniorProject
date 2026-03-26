#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_full_pair_%j.log
#SBATCH --job-name=deeppass_fpair

# Full lm-eval (NO subsample) on best pair (0,7)+(45,52)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Full lm-eval pair (0,7)+(45,52) ==="
echo "Started: $(date)"

$PYTHON scripts/experiments/lm_eval_runtime_dup.py \
    --model models/full/calme-2.1-qwen2-72b \
    --blocks "0,7;45,52" \
    --tasks leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro \
    --limit 0 \
    --output results/data/72b/lm_eval/full/pair_0_7_45_52.json

echo "=== Done at $(date) ==="
