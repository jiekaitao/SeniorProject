#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_paradigm_ll3v2_%j.log
#SBATCH --job-name=ps_ll3v2

# Paradigm Shift v2: LLaMA 3 8B — FIXED
# v1 diagnosis: LR too high (5e-5), LoRA too aggressive on FFN
# - EQ-bench dropped 59.5 -> 44.0 because LoRA over-optimized CE
# - Math stayed flat (0.64 both times)
# Fix: 10x lower LR, half the steps, minimal FFN rank

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Paradigm Shift v2: LLaMA 3 8B ===" && echo "Started: $(date)"

envs/deeppass/bin/python sirt/paradigm_shift.py \
    --model /data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf \
    --name llama3_8b_v2 \
    --core_start 10 --core_end 13 \
    --rank_attn 8 --rank_ffn 1 \
    --max_steps 150 --lr 5e-6 --warmup_steps 40 \
    --gate_weight 0.10 --oplora_weight 0.02 \
    --ffn_whisper_beta 0.2

echo "=== Finished: $(date) ==="
