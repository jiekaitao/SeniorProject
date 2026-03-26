#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_seam_72b_%j.log
#SBATCH --job-name=deeppass_seam72b

# SVD subspace patching + gated residual on 72B
# (norm-preserving is running separately)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Seam experiments on 72B ==="
echo "Started: $(date)"

echo ""
echo "=========================================="
echo "EXPERIMENT 1: SVD Subspace Patching (72B)"
echo "=========================================="
$PYTHON -c "
import scripts.experiments.spectral.svd_subspace_patching as svd
svd.MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
svd.RESULTS_PATH = 'results/data/72b/seam/svd_subspace_72b.json'
import os; os.makedirs('results/data/72b/seam', exist_ok=True)
svd.main()
"

echo ""
echo "=========================================="
echo "EXPERIMENT 2: Gated Residual (72B)"
echo "=========================================="
$PYTHON -c "
from pathlib import Path
import scripts.experiments.spectral.gated_residual_test as gated
gated.MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
gated.RESULTS_PATH = Path('results/data/72b/seam/gated_residual_72b.json')
import os; os.makedirs('results/data/72b/seam', exist_ok=True)
gated.main()
"

echo ""
echo "=== All 72B seam experiments done at $(date) ==="
