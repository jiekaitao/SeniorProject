#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_cohere_35b_%j.log
#SBATCH --job-name=coh_35b
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Cohere Command R 35B ===" && echo "Started: $(date)"
envs/deeppass/bin/python scripts/experiments/universal_sweep.py \
    --model CohereForAI/c4ai-command-r-v01 --name cohere_35b
echo "=== Finished: $(date) ==="
