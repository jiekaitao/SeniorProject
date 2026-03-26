#!/bin/bash
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass

LOG="/blue/cis4914/jietao/DeepPass/results/junction_ft_v2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

# Wait for the v1 junction FT (comprehensive) to finish
echo "Waiting for v1 experiments to finish GPU work..."
while pgrep -f "comprehensive_junction_ft" > /dev/null 2>&1; do
    sleep 30
done
# Also wait if lm-eval started already
while pgrep -f "lm_eval" > /dev/null 2>&1; do
    echo "  lm-eval running, waiting..."
    sleep 120
done

echo "GPU free! Starting junction FT v2 at $(date)"
python -u /blue/cis4914/jietao/DeepPass/scripts/junction_ft_v2.py 2>&1
echo "Junction FT v2 complete at $(date)"
