#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_paradigm_lmeval_ll3_%j.log
#SBATCH --job-name=ps_eval

# Full lm-eval on Paradigm Shift LLaMA 3 8B
# Runs BBH, MATH-Hard, MMLU-Pro, MuSR (full dataset)
# Also runs baseline for comparison
# Submit AFTER paradigm_llama3.sh completes

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== lm-eval: Paradigm Shift LLaMA 3 8B ===" && echo "Started: $(date)"

CKPT=sirt/recursion_ft/llama3_8b_paradigm/checkpoint.pt

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found at $CKPT"
    echo "Run paradigm_llama3.sh first"
    exit 1
fi

# Full eval with baseline comparison
envs/deeppass/bin/python sirt/paradigm_lmeval.py \
    --model /data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf \
    --checkpoint "$CKPT" \
    --tasks bbh,math,mmlu_pro,musr \
    --limit 0.0 \
    --baseline \
    --ffn_whisper 0.0

# Also with FFN whisper
echo "--- Now with FFN whisper ---"
envs/deeppass/bin/python sirt/paradigm_lmeval.py \
    --model /data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf \
    --checkpoint "$CKPT" \
    --tasks bbh,math,mmlu_pro,musr \
    --limit 0.0 \
    --ffn_whisper 0.2 \
    --output sirt/recursion_ft/llama3_8b_paradigm/lmeval_whisper.json

echo "=== Finished: $(date) ==="
