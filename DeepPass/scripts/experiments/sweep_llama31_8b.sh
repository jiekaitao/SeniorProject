#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_llama31_8b_%j.log
#SBATCH --job-name=ll31_8b
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== LLaMA 3.1 8B ===" && echo "Started: $(date)"
envs/deeppass/bin/python scripts/experiments/universal_sweep.py \
    --model meta-llama/Llama-3.1-8B-Instruct --name llama31_8b
echo "=== Finished: $(date) ==="
