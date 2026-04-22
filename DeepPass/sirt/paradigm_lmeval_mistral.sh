#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_paradigm_lmeval_mis_%j.log
#SBATCH --job-name=ps_ev_m

# Full lm-eval on Paradigm Shift Mistral 7B
# Submit AFTER paradigm_mistral.sh completes

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== lm-eval: Paradigm Shift Mistral 7B ===" && echo "Started: $(date)"

# Use v2 checkpoint (core [28,29) — the working config)
CKPT=sirt/recursion_ft/mistral_7b_v2_paradigm/checkpoint.pt

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found at $CKPT"
    echo "Run paradigm_mistral_v2.sh first"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0

# Full eval with baseline comparison
envs/deeppass/bin/python sirt/paradigm_lmeval.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --checkpoint "$CKPT" \
    --tasks bbh,math,mmlu_pro,musr \
    --limit 0.0 \
    --baseline \
    --ffn_whisper 0.0

# Also with FFN whisper
echo "--- Now with FFN whisper ---"
envs/deeppass/bin/python sirt/paradigm_lmeval.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --checkpoint "$CKPT" \
    --tasks bbh,math,mmlu_pro,musr \
    --limit 0.0 \
    --ffn_whisper 0.2 \
    --output sirt/recursion_ft/mistral_7b_v2_paradigm/lmeval_whisper.json

echo "=== Finished: $(date) ==="
