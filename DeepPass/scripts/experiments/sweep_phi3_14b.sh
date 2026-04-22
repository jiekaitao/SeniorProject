#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_phi3_14b_%j.log
#SBATCH --job-name=phi3_14
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Phi-3 Medium 14B ===" && echo "Started: $(date)"
envs/deeppass/bin/python scripts/experiments/universal_sweep.py \
    --model microsoft/Phi-3-medium-4k-instruct --name phi3_14b
echo "=== Finished: $(date) ==="
