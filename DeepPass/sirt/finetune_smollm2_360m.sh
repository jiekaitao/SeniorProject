#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_smollm2_360m_%j.log
#SBATCH --job-name=sm2_360

# SmolLM2-360M: SOTA sub-500M model (trained on 4T tokens)
# Apply our full DeepPass analysis: SBUID screen + Tier 2 + sublayer

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== SmolLM2-360M Analysis ==="
echo "Started: $(date)"

$PYTHON scripts/experiments/universal_sweep.py \
    --model HuggingFaceTB/SmolLM2-360M-Instruct --name smollm2_360m --max_blocks 12

echo "=== Finished: $(date) ==="
