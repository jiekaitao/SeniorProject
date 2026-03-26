#!/bin/bash
# DeepPass Overnight Runner — Full experiment suite
# Sequences GPU-intensive tasks to avoid VRAM conflicts
set -e

module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass

SCRIPTS="/blue/cis4914/jietao/DeepPass/scripts"
RESULTS="/blue/cis4914/jietao/DeepPass/results"
MODELS="/blue/cis4914/jietao/DeepPass/models"
LOG="$RESULTS/overnight_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo "DeepPass Overnight Runner"
echo "Started: $(date)"
echo "============================================"

# ============================================
# PHASE 1: Full Leaderboard Benchmarks on 72B
# ============================================
echo ""
echo "=== PHASE 1: Leaderboard Benchmarks ==="

# 1a. Baseline — standard lm-eval (KV cache works)
echo "--- 1a. Baseline lm-eval (6 benchmarks) ---"
echo "Started: $(date)"
python -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODELS/full/calme-2.1-qwen2-72b,dtype=bfloat16,trust_remote_code=True" \
    --tasks leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_gpqa,leaderboard_musr,leaderboard_mmlu_pro \
    --batch_size auto \
    --output_path "$RESULTS/lm_eval_baseline_72b" \
    --device cuda \
    --num_fewshot 0 \
    2>&1 || echo "lm-eval baseline failed, continuing..."
echo "Baseline lm-eval finished: $(date)"

# 1b. Duplicated model — needs the saved checkpoint
DUP_MODEL="$MODELS/full/calme-2.1-qwen2-72b-dup-45-52"
if [ -d "$DUP_MODEL" ] && [ -f "$DUP_MODEL/config.json" ]; then
    echo "--- 1b. Duplicated (45,52) lm-eval (6 benchmarks) ---"
    echo "Started: $(date)"
    python -m lm_eval \
        --model hf \
        --model_args "pretrained=$DUP_MODEL,dtype=bfloat16,trust_remote_code=True" \
        --tasks leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_gpqa,leaderboard_musr,leaderboard_mmlu_pro \
        --batch_size auto \
        --output_path "$RESULTS/lm_eval_dup45_52_72b" \
        --device cuda \
        --num_fewshot 0 \
        2>&1 || echo "lm-eval duplicated failed, continuing..."
    echo "Duplicated lm-eval finished: $(date)"
else
    echo "WARNING: Duplicated model not found at $DUP_MODEL, skipping..."
fi

# ============================================
# PHASE 2: Spectral Analysis on 7B (fast)
# ============================================
echo ""
echo "=== PHASE 2: Spectral Analysis (7B) ==="
echo "Started: $(date)"
python "$SCRIPTS/spectral_analysis.py" \
    --model "$MODELS/small/Qwen2-7B-Instruct" \
    --step 1 \
    --block-sizes "1,2,3,4,5,6,7,8,9,10" \
    --output "$RESULTS/spectral_Qwen2-7B-Instruct" \
    2>&1 || echo "Spectral 7B failed, continuing..."
echo "Spectral 7B finished: $(date)"

# ============================================
# PHASE 3: Brain Scanner on 7B (full sweep)
# ============================================
echo ""
echo "=== PHASE 3: Brain Scanner (7B) ==="
echo "Started: $(date)"
python "$SCRIPTS/brain_scanner.py" \
    --model "$MODELS/small/Qwen2-7B-Instruct" \
    --step 2 \
    --max-dup 14 \
    --output "$RESULTS/sweep_Qwen2-7B-Instruct" \
    2>&1 || echo "Brain scanner 7B failed, continuing..."
echo "Brain scanner 7B finished: $(date)"

# ============================================
# PHASE 4: Spectral Analysis on 72B (the prize)
# ============================================
echo ""
echo "=== PHASE 4: Spectral Analysis (72B) ==="
echo "Started: $(date)"
python "$SCRIPTS/spectral_analysis.py" \
    --model "$MODELS/full/calme-2.1-qwen2-72b" \
    --step 2 \
    --block-sizes "5,6,7,8,9,10" \
    --output "$RESULTS/spectral_calme-72b" \
    2>&1 || echo "Spectral 72B failed, continuing..."
echo "Spectral 72B finished: $(date)"

# ============================================
# PHASE 5: Multi-pass experiments on 7B
# ============================================
echo ""
echo "=== PHASE 5: Multi-pass (3x duplication) on 7B ==="
echo "Started: $(date)"
python "$SCRIPTS/multi_pass_test.py" \
    --model "$MODELS/small/Qwen2-7B-Instruct" \
    2>&1 || echo "Multi-pass test failed, continuing..."

echo ""
echo "============================================"
echo "Overnight Runner Complete"
echo "Finished: $(date)"
echo "Results in: $RESULTS/"
echo "============================================"
