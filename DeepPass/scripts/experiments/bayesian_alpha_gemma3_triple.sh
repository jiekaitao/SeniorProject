#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_bayesian_alpha_g3_%j.log
#SBATCH --job-name=g3_alpha

# Bayesian alpha optimization for Gemma3-27B triple (0,2)+(12,13)+(47,48)
# 2 parallel workers (1 GPU each), 20 trials per worker = 40 total
# Then validates top 5 with full probes
# Estimated: ~50 min search + ~20 min validation = ~70 min total

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python
SCRIPT=scripts/experiments/bayesian_alpha_gemma3_triple.py

echo "=== Bayesian Alpha Optimization: Gemma3-27B Triple ==="
echo "Blocks: (0,2)+(12,13)+(47,48)"
echo "Started: $(date)"
echo "GPUs: $(nvidia-smi -L)"

mkdir -p results/data/gemma3_27b/bayesian_alpha_triple

# Launch 2 workers in parallel, each on its own GPU
echo ""
echo "--- Launching Worker 0 (GPU 0) ---"
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --worker 0 --n-trials 20 &
PID0=$!

# Stagger start by 90s to avoid simultaneous model loading (CPU RAM spike)
echo "Waiting 90s before launching Worker 1..."
sleep 90

echo "--- Launching Worker 1 (GPU 1) ---"
CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT --worker 1 --n-trials 20 &
PID1=$!

echo "Workers running: PID0=$PID0 PID1=$PID1"

# Wait for both workers to finish
wait $PID0
STATUS0=$?
echo "Worker 0 finished with status $STATUS0"

wait $PID1
STATUS1=$?
echo "Worker 1 finished with status $STATUS1"

# Validate top configs with full probes on GPU 0
echo ""
echo "=== Validation Phase ==="
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --validate

echo ""
echo "=== COMPLETE ==="
echo "Finished: $(date)"
