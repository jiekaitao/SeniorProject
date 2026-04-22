#!/bin/bash
#SBATCH --job-name=bbh
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_bbh_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

# Try BBH (Big-Bench Hard) - MCQA-compatible subtasks
echo "=== BBH: reasoning benchmark ==="
$PYTHON -c "
from datasets import load_dataset
# Check what BBH subtasks exist
try:
    ds = load_dataset('lukaemon/bbh', 'logical_deduction_three_objects')
    print(f'BBH 3obj: {len(ds[\"test\"])} samples')
    print(f'Keys: {list(ds[\"test\"][0].keys())}')
    print(f'Sample: {ds[\"test\"][0]}')
except Exception as e:
    print(f'Failed: {e}')
"
