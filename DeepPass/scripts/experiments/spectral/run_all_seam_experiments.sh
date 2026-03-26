#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_seam_experiments_%j.log
#SBATCH --job-name=deeppass_seam_exp

cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Starting all seam experiments on Qwen3.5-9B ==="
echo "Started: $(date)"

echo ""
echo "=========================================="
echo "EXPERIMENT 1: SVD Subspace Patching"
echo "=========================================="
$PYTHON scripts/experiments/spectral/svd_subspace_patching.py

echo ""
echo "=========================================="
echo "EXPERIMENT 2: Norm-Preserving Projection"
echo "=========================================="
$PYTHON scripts/experiments/spectral/norm_preserving_test.py

echo ""
echo "=========================================="
echo "EXPERIMENT 3: Gated Residual (Analytical)"
echo "=========================================="
$PYTHON scripts/experiments/spectral/gated_residual_test.py

echo ""
echo "=== All seam experiments done at $(date) ==="
