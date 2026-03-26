#!/bin/bash
# DeepPass: Comprehensive overnight experiment suite
# Run after lm-eval baseline completes (or in sequence)
set -e

module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass

S="/blue/cis4914/jietao/DeepPass/scripts"
R="/blue/cis4914/jietao/DeepPass/results"
M="/blue/cis4914/jietao/DeepPass/models"
LOG="$R/experiments_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "DeepPass Experiment Suite — $(date)"
echo "=========================================="

# ---- EXPERIMENT 1: Save duplicated 72B model ----
echo ""
echo "=== EXP 1: Save duplicated model ==="
python "$S/save_duplicated_model.py" 2>&1 || echo "WARN: save failed"

# ---- EXPERIMENT 2: lm-eval on duplicated model ----
DUP_MODEL="$M/full/calme-2.1-qwen2-72b-dup-45-52"
if [ -d "$DUP_MODEL" ] && [ -f "$DUP_MODEL/config.json" ]; then
    echo ""
    echo "=== EXP 2: lm-eval duplicated (45,52) ==="
    python -m lm_eval \
        --model hf \
        --model_args "pretrained=$DUP_MODEL,dtype=bfloat16,trust_remote_code=True" \
        --tasks leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_gpqa,leaderboard_musr,leaderboard_mmlu_pro \
        --batch_size auto \
        --output_path "$R/lm_eval_dup45_52_72b" \
        --device cuda 2>&1 || echo "WARN: lm-eval dup failed"
fi

# ---- EXPERIMENT 3: Spectral analysis 7B ----
echo ""
echo "=== EXP 3: Spectral analysis (7B) ==="
python "$S/spectral_analysis.py" \
    --model "$M/small/Qwen2-7B-Instruct" \
    --step 1 \
    --block-sizes "1,2,3,4,5,6,7,8,9,10" \
    --output "$R/spectral_7B" 2>&1 || echo "WARN: spectral 7B failed"

# ---- EXPERIMENT 4: Brain scanner 7B ----
echo ""
echo "=== EXP 4: Brain scanner (7B, step=2) ==="
python "$S/brain_scanner.py" \
    --model "$M/small/Qwen2-7B-Instruct" \
    --step 2 \
    --max-dup 14 \
    --output "$R/sweep_7B" 2>&1 || echo "WARN: brain scanner failed"

# ---- EXPERIMENT 5: Multi-pass test 7B ----
echo ""
echo "=== EXP 5: Multi-pass (1-5x) on 7B ==="
python "$S/multi_pass_test.py" \
    --model "$M/small/Qwen2-7B-Instruct" 2>&1 || echo "WARN: multi-pass failed"

# ---- EXPERIMENT 6: Junction fine-tuning 7B ----
echo ""
echo "=== EXP 6: Junction fine-tuning (7B) ==="
python "$S/junction_finetune.py" \
    --model "$M/small/Qwen2-7B-Instruct" \
    --steps 200 \
    --lr 1e-5 2>&1 || echo "WARN: junction FT failed"

# ---- EXPERIMENT 7: Spectral analysis 72B (selective) ----
echo ""
echo "=== EXP 7: Spectral analysis (72B, blocks 5-10) ==="
python "$S/spectral_analysis.py" \
    --model "$M/full/calme-2.1-qwen2-72b" \
    --step 2 \
    --block-sizes "5,6,7,8,9,10" \
    --output "$R/spectral_72B" 2>&1 || echo "WARN: spectral 72B failed"

# ---- EXPERIMENT 8: Validate spectral predictions on 7B ----
echo ""
echo "=== EXP 8: Validate spectral predictions vs brain scan ==="
python "$S/validate_spectral.py" \
    --spectral "$R/spectral_7B/spectral_results.json" \
    --sweep "$R/sweep_7B/sweep_results.json" \
    --output "$R/validation_7B" 2>&1 || echo "WARN: validation failed (script may not exist yet)"

echo ""
echo "=========================================="
echo "All experiments complete — $(date)"
echo "=========================================="
