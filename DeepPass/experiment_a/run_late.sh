#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_expa_late_%j.log
#SBATCH --job-name=expa
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Exp A LATE (layers 20-27) ==="
envs/deeppass/bin/python experiment_a/train.py 20,21,22,23,24,25,26,27
exit 0
