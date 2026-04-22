#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_mistral_7b_%j.log
#SBATCH --job-name=mis_7b
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Mistral 7B ===" && echo "Started: $(date)"
envs/deeppass/bin/python scripts/experiments/universal_sweep.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 --name mistral_7b
echo "=== Finished: $(date) ==="
