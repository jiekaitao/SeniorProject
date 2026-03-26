#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_gated_residual_%j.log
#SBATCH --job-name=deeppass_gated_res

cd /blue/cis4914/jietao/DeepPass
/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python scripts/experiments/spectral/gated_residual_test.py
