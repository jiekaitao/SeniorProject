#!/bin/bash
# Run baseline lm-eval after all other GPU work finishes
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass

R="/blue/cis4914/jietao/DeepPass/results"
M="/blue/cis4914/jietao/DeepPass/models"

# Wait for the experiment chain to finish
echo "Waiting for experiment chain to finish..."
while pgrep -f "run_after_baseline" > /dev/null 2>&1; do
    echo "  chain still running... $(date)"
    sleep 120
done
echo "Chain done! Starting baseline lm-eval at $(date)"

python -m lm_eval \
    --model hf \
    --model_args "pretrained=$M/full/calme-2.1-qwen2-72b,dtype=bfloat16,trust_remote_code=True" \
    --tasks leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_gpqa,leaderboard_musr,leaderboard_mmlu_pro \
    --batch_size auto \
    --output_path "$R/lm_eval_baseline_72b" \
    --device cuda 2>&1

echo "Baseline lm-eval complete at $(date)"

# Compile all results
python /blue/cis4914/jietao/DeepPass/scripts/compile_results.py 2>&1
