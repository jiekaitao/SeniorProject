#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ksweep_mis_%j.log
#SBATCH --job-name=ks_mis

# K-Degradation Sweep: Mistral 7B
# Best single block: (28,29) — the +3.50 winner
# Test K=1..4 x {full, attn_only, whisper}

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== K-Degradation: Mistral 7B ===" && echo "Started: $(date)"

envs/deeppass/bin/python sirt/k_degradation_sweep.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --name mistral_7b \
    --core_start 28 --core_end 29 \
    --max_k 4

echo "=== Finished: $(date) ==="
