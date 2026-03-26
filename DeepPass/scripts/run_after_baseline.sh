#!/bin/bash
# Wait for baseline lm-eval to finish, then run all remaining experiments
set -e

module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass

S="/blue/cis4914/jietao/DeepPass/scripts"
R="/blue/cis4914/jietao/DeepPass/results"
M="/blue/cis4914/jietao/DeepPass/models"
LOG="$R/after_baseline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "Waiting for lm-eval baseline to finish..."
# Wait for lm-eval to finish by checking for the python -m lm_eval process
while pgrep -f "python.*lm_eval" > /dev/null 2>&1; do
    echo "  lm-eval still running... $(date)"
    sleep 120
done
echo "Baseline lm-eval done! Starting experiments at $(date)"

# 1. Save duplicated model (deep copy)
echo ""
echo "=== Saving duplicated model ==="
python "$S/save_duplicated_model.py" 2>&1 || echo "WARN: save failed"

# 2. lm-eval on duplicated model
DUP_MODEL="$M/full/calme-2.1-qwen2-72b-dup-45-52"
if [ -d "$DUP_MODEL" ] && [ -f "$DUP_MODEL/config.json" ]; then
    echo ""
    echo "=== lm-eval duplicated ==="
    python -m lm_eval \
        --model hf \
        --model_args "pretrained=$DUP_MODEL,dtype=bfloat16,trust_remote_code=True" \
        --tasks leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_gpqa,leaderboard_musr,leaderboard_mmlu_pro \
        --batch_size auto \
        --output_path "$R/lm_eval_dup45_52_72b" \
        --device cuda 2>&1 || echo "WARN: dup lm-eval failed"
fi

# 3. Spectral analysis on 7B
echo ""
echo "=== Spectral 7B ==="
python "$S/spectral_analysis.py" \
    --model "$M/small/Qwen2-7B-Instruct" \
    --step 1 \
    --block-sizes "1,2,3,4,5,6,7,8,9,10" \
    --output "$R/spectral_7B" 2>&1 || echo "WARN: spectral 7B failed"

# 4. Brain scanner 7B
echo ""
echo "=== Brain scanner 7B ==="
python "$S/brain_scanner.py" \
    --model "$M/small/Qwen2-7B-Instruct" \
    --step 2 \
    --max-dup 14 \
    --output "$R/sweep_7B" 2>&1 || echo "WARN: brain scanner failed"

# 5. Validate spectral vs sweep
echo ""
echo "=== Validate spectral predictions ==="
python "$S/validate_spectral.py" \
    --spectral "$R/spectral_7B/spectral_results.json" \
    --sweep "$R/sweep_7B/sweep_results.json" \
    --output "$R/validation_7B" 2>&1 || echo "WARN: validation failed"

# 6. Multi-pass 7B
echo ""
echo "=== Multi-pass 7B ==="
python "$S/multi_pass_test.py" \
    --model "$M/small/Qwen2-7B-Instruct" 2>&1 || echo "WARN: multi-pass failed"

# 7. Junction fine-tuning 7B
echo ""
echo "=== Junction fine-tuning 7B ==="
python "$S/junction_finetune.py" \
    --model "$M/small/Qwen2-7B-Instruct" \
    --steps 200 2>&1 || echo "WARN: junction FT failed"

# 8. Spectral 72B
echo ""
echo "=== Spectral 72B ==="
python "$S/spectral_analysis.py" \
    --model "$M/full/calme-2.1-qwen2-72b" \
    --step 3 \
    --block-sizes "5,6,7,8,9,10" \
    --output "$R/spectral_72B" 2>&1 || echo "WARN: spectral 72B failed"

# 9. DeepPass unified analysis on 7B
echo ""
echo "=== DeepPass Analysis 7B ==="
python "$S/deeppass_analysis.py" \
    --model "$M/small/Qwen2-7B-Instruct" \
    --step 1 \
    --block-sizes "3,4,5,6,7,8,9,10" \
    --top-k 15 \
    --output "$R/deeppass_7B" 2>&1 || echo "WARN: deeppass 7B failed"

# 10. DeepPass analysis on 72B (targeted blocks near Ng's optimal)
echo ""
echo "=== DeepPass Analysis 72B ==="
python "$S/deeppass_analysis.py" \
    --model "$M/full/calme-2.1-qwen2-72b" \
    --step 2 \
    --block-sizes "5,6,7,8,9,10" \
    --top-k 10 \
    --output "$R/deeppass_72B" 2>&1 || echo "WARN: deeppass 72B failed"

# 11. Compile all results
echo ""
echo "=== Compiling Results ==="
python "$S/compile_results.py" 2>&1

echo ""
echo "=========================================="
echo "All experiments complete — $(date)"
echo "=========================================="
