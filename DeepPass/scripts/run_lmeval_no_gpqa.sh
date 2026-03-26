#!/bin/bash
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass

R="/blue/cis4914/jietao/DeepPass/results"
M="/blue/cis4914/jietao/DeepPass/models"
LOG="$R/lmeval_no_gpqa_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

# Wait for 72B experiments to finish (spectral search + junction FT)
echo "Waiting for 72B experiments..."
while pgrep -f "spectral_guided_search\|junction_ft_72b" > /dev/null 2>&1; do
    sleep 60
done
echo "72B experiments done! Starting lm-eval at $(date)"

# Run 5 benchmarks (skip gated GPQA)
TASKS="leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro"

echo "=== BASELINE ==="
python -m lm_eval \
    --model hf \
    --model_args "pretrained=$M/full/calme-2.1-qwen2-72b,dtype=bfloat16,trust_remote_code=True" \
    --tasks $TASKS \
    --batch_size auto \
    --output_path "$R/lm_eval_baseline_72b" \
    --device cuda 2>&1
echo "=== BASELINE DONE === $(date)"

echo "=== DUPLICATED ==="
python -m lm_eval \
    --model hf \
    --model_args "pretrained=$M/full/calme-2.1-qwen2-72b-dup-45-52,dtype=bfloat16,trust_remote_code=True" \
    --tasks $TASKS \
    --batch_size auto \
    --output_path "$R/lm_eval_dup45_52_72b" \
    --device cuda 2>&1
echo "=== DUPLICATED DONE === $(date)"

python /blue/cis4914/jietao/DeepPass/scripts/compile_results.py 2>&1
