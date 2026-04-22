#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_gemma2_9b_%j.log
#SBATCH --job-name=g2_9b
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Gemma 2 9B ===" && echo "Started: $(date)"
envs/deeppass/bin/python scripts/experiments/universal_sweep.py \
    --model google/gemma-2-9b-it --name gemma2_9b
echo "=== Finished: $(date) ==="
