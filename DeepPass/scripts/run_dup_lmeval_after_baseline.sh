#!/bin/bash
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass

R="/blue/cis4914/jietao/DeepPass/results"
M="/blue/cis4914/jietao/DeepPass/models"
LOG="$R/dup_lmeval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "Waiting for baseline lm-eval to finish..."
while pgrep -f "lm_eval" > /dev/null 2>&1; do
    sleep 120
done
echo "Baseline done! Starting duplicated lm-eval at $(date)"

python -m lm_eval \
    --model hf \
    --model_args "pretrained=$M/full/calme-2.1-qwen2-72b-dup-45-52,dtype=bfloat16,trust_remote_code=True" \
    --tasks leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_gpqa,leaderboard_musr,leaderboard_mmlu_pro \
    --batch_size auto \
    --output_path "$R/lm_eval_dup45_52_72b" \
    --device cuda 2>&1

echo "Duplicated lm-eval complete at $(date)"
python /blue/cis4914/jietao/DeepPass/scripts/compile_results.py 2>&1
