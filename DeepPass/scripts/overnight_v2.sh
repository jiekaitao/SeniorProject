#!/bin/bash
# Overnight Runner V2 — Three tasks for overnight execution
# Phase 1: (45,52) lm-eval at 15% subsample (~5 hours)
# Phase 2: Junction FT V3 on 72B (50,60) — two-stage memory-efficient (~1 hour)
# Phase 3: Full evaluation — baseline + (50,60), one task at a time (as far as time allows)
#
# Results saved per-phase so partial completion is still useful.
# No set -e: phases should continue even if one fails (piped through tee).

# Environment
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass

cd /blue/cis4914/jietao/DeepPass

TASKS="leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro"

echo "=============================================="
echo "OVERNIGHT V2 — Started at $(date)"
echo "=============================================="

###############################################################################
# PHASE 1: (45,52) lm-eval at 15% subsample
###############################################################################
echo ""
echo "====== PHASE 1: (45,52) lm-eval at 15% ======"
echo "Started at $(date)"

python -m lm_eval \
    --model hf \
    --model_args "pretrained=models/full/calme-2.1-qwen2-72b-dup-45-52,dtype=bfloat16,trust_remote_code=True" \
    --tasks $TASKS \
    --batch_size auto \
    --limit 0.15 \
    --output_path results/lm_eval_dup45_52_72b_15pct \
    --device cuda 2>&1 | tee results/lmeval_dup45_52.log

echo "PHASE 1 COMPLETE at $(date)"

###############################################################################
# PHASE 2: Junction FT V3 on 72B (50,60)
###############################################################################
echo ""
echo "====== PHASE 2: Junction FT V3 on 72B (50,60) ======"
echo "Started at $(date)"

python scripts/junction_ft_v3_72b.py 2>&1 | tee results/junction_ft_v3_72b.log

echo "PHASE 2 COMPLETE at $(date)"

###############################################################################
# PHASE 3: Full evaluation — one task at a time
# Run baseline and (50,60) for each task sequentially.
# Each completed task is independently saved, so partial completion is useful.
###############################################################################
echo ""
echo "====== PHASE 3: Full evaluation (no limit) ======"
echo "Started at $(date)"

for task in leaderboard_ifeval leaderboard_bbh leaderboard_math_hard leaderboard_musr leaderboard_mmlu_pro; do
    echo ""
    echo "--- Full eval: BASELINE — $task ($(date)) ---"
    python -m lm_eval \
        --model hf \
        --model_args "pretrained=models/full/calme-2.1-qwen2-72b,dtype=bfloat16,trust_remote_code=True" \
        --tasks $task \
        --batch_size auto \
        --output_path results/lm_eval_baseline_72b_full \
        --device cuda 2>&1 | tee -a results/lmeval_baseline_full.log

    echo "--- Full eval: (50,60) — $task ($(date)) ---"
    python -m lm_eval \
        --model hf \
        --model_args "pretrained=models/full/calme-2.1-qwen2-72b-dup-50-60,dtype=bfloat16,trust_remote_code=True" \
        --tasks $task \
        --batch_size auto \
        --output_path results/lm_eval_dup50_60_72b_full \
        --device cuda 2>&1 | tee -a results/lmeval_dup50_60_full.log
done

echo ""
echo "=============================================="
echo "OVERNIGHT V2 — Completed at $(date)"
echo "=============================================="
