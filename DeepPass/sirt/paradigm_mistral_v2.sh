#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_paradigm_misv2_%j.log
#SBATCH --job-name=ps_misv2

# Paradigm Shift v2: Mistral 7B — FIXED
# v1 diagnosis: core [27,30) catastrophically broke model (pre-ft K=2 = 0.20)
# Previous best single block was (28,29)
# Fix: use core [28,29) — just 1 layer, the known-good block

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Paradigm Shift v2: Mistral 7B ===" && echo "Started: $(date)"

envs/deeppass/bin/python sirt/paradigm_shift.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --name mistral_7b_v2 \
    --core_start 28 --core_end 29 \
    --rank_attn 8 --rank_ffn 1 \
    --max_steps 200 --lr 1e-5 --warmup_steps 50 \
    --gate_weight 0.10 --oplora_weight 0.02 \
    --ffn_whisper_beta 0.2

echo "=== Finished: $(date) ==="
