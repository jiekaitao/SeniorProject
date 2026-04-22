#!/bin/bash
#SBATCH --job-name=cr_multi
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_creative_multitask_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Multi-task controller (ONE controller, ALL SpatialEval) ==="
$PYTHON solver/eval_deliberation_creative.py --experiment multitask --seeds 42,7 --steps 4000

echo ""
echo "=== Progressive curriculum (1→2→3→5 rounds) ==="
$PYTHON solver/eval_deliberation_creative.py --experiment curriculum --seeds 42,7 --steps 3000
