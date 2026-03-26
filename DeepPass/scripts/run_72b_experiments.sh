#!/bin/bash
# Run all 72B experiments after lm-eval finishes
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass

S="/blue/cis4914/jietao/DeepPass/scripts"
R="/blue/cis4914/jietao/DeepPass/results"
LOG="$R/72b_experiments_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "Waiting for lm-eval to finish..."
while pgrep -f "lm_eval" > /dev/null 2>&1; do
    sleep 120
done
echo "lm-eval done! Starting 72B experiments at $(date)"

# 1. Spectral-guided search (find better config than Ng's 45,52)
echo ""
echo "=== Spectral-Guided Search 72B ==="
python "$S/spectral_guided_search_72b.py" 2>&1

# 2. Junction fine-tuning on 72B
echo ""
echo "=== Junction Fine-Tuning 72B ==="
python "$S/junction_ft_72b.py" 2>&1

# 3. Final compilation
echo ""
echo "=== Final Results ==="
python "$S/compile_results.py" 2>&1

echo ""
echo "All 72B experiments complete at $(date)"
