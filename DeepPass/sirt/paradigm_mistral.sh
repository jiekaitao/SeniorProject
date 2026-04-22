#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_paradigm_mis_%j.log
#SBATCH --job-name=ps_mis

# Paradigm Shift: Mistral 7B
# THE model where raw duplication already improved lm-eval
# Core [27,30) — best block around (28,29)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Paradigm Shift: Mistral 7B ===" && echo "Started: $(date)"

envs/deeppass/bin/python sirt/paradigm_shift.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --name mistral_7b \
    --core_start 27 --core_end 30 \
    --rank_attn 8 --rank_ffn 4 \
    --max_steps 320 --lr 5e-5 --warmup_steps 80 \
    --gate_weight 0.10 --oplora_weight 0.02 \
    --ffn_whisper_beta 0.2

echo "=== Finished: $(date) ==="
