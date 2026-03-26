#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_oracle_triple_%j.log
#SBATCH --job-name=deeppass_oracle_triple

cd /blue/cis4914/jietao/DeepPass

PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Job started at $(date) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader
echo ""

echo "=== Oracle Seam Patching (72B) ==="
$PYTHON scripts/experiments/spectral/oracle_seam_patching.py
echo ""

echo "=== Triple Stacking (72B) ==="
$PYTHON scripts/experiments/spectral/triple_stacking_72b.py
echo ""

echo "=== All done at $(date) ==="
