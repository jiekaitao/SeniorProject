#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_expa_%j.log
#SBATCH --job-name=expa

# Experiment A: Band-Recurrent Llama 3.1 8B
# Frozen 8B base + attention replay on layers 12-15 with LoRA rank 16
# Hard-token routing (top 25% entropy), mixed K=1/K=2 training
# ~5M trainable params on 8B frozen base
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Experiment A: Band-Recurrent Llama ===" && echo "Started: $(date)"

envs/deeppass/bin/python experiment_a/train.py

echo "=== Finished: $(date) ==="
