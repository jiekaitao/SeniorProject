#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_reason_mis_attn_%j.log
#SBATCH --job-name=reas_a

# Reasoning probe on Mistral with ATTENTION-ONLY duplication
# Since the K-sweep showed attn-only is stable, test if it helps on trick questions
# Uses the same reasoning_probe.py but manually patches to attn-only

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== K-sweep: Mistral attention-only deeper ===" && echo "Started: $(date)"

# Run K-sweep with a wider block range to find better blocks
envs/deeppass/bin/python sirt/k_degradation_sweep.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --name mistral_7b_block15_16 \
    --core_start 15 --core_end 16 \
    --max_k 4

echo "=== Finished: $(date) ==="
