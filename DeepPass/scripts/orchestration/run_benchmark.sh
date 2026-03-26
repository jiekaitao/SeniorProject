#!/bin/bash
#SBATCH --job-name=deeppass-bench
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/benchmark_%j.log
#SBATCH --error=/blue/cis4914/jietao/DeepPass/results/benchmark_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8

# DeepPass Benchmark Runner
# Benchmarks a model before and after Ng's layer duplication (i=45, j=52)

module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass

SCRIPTS_DIR="/blue/cis4914/jietao/DeepPass/scripts"
RESULTS_DIR="/blue/cis4914/jietao/DeepPass/results"

# Default model — change this to test different models
MODEL_PATH="${1:-/blue/cis4914/jietao/DeepPass/models/full/calme-2.1-qwen2-72b}"

# Duplication config — Ng's optimal for Qwen2-72B
DUP_I="${2:-45}"
DUP_J="${3:-52}"

echo "============================================"
echo "DeepPass Benchmark"
echo "Model: ${MODEL_PATH}"
echo "Duplication: (${DUP_I}, ${DUP_J})"
echo "Started: $(date)"
echo "============================================"

# Run baseline
echo ""
echo "--- BASELINE (no duplication) ---"
python ${SCRIPTS_DIR}/benchmark.py \
    --model "${MODEL_PATH}" \
    --baseline-only \
    --tasks math_probe \
    --tag baseline

# Run with layer duplication
echo ""
echo "--- DUPLICATED (${DUP_I}, ${DUP_J}) ---"
python ${SCRIPTS_DIR}/benchmark.py \
    --model "${MODEL_PATH}" \
    --i ${DUP_I} \
    --j ${DUP_J} \
    --tasks math_probe \
    --tag "dup_${DUP_I}_${DUP_J}"

echo ""
echo "Completed: $(date)"
