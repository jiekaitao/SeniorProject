#!/bin/bash
# Run oracle seam patching, then triple stacking sequentially
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python
cd /blue/cis4914/jietao/DeepPass

echo "=== Starting Oracle Seam Patching ==="
$PYTHON scripts/experiments/spectral/oracle_seam_patching.py 2>&1 | tee results/72b_oracle_seam_patching.log

echo ""
echo "=== Starting Triple Stacking ==="
$PYTHON scripts/experiments/spectral/triple_stacking_72b.py 2>&1 | tee results/72b_triple_stacking.log

echo ""
echo "=== ALL DONE ==="
