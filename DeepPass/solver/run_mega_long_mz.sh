#!/bin/bash
#SBATCH --job-name=long_mz
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_mega_long_mz_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python
# This job runs 4 configs but switches between base and instruct — use base model for first 2
$PYTHON solver/mega_runner.py \
    --model models/full/Llama-3.1-8B \
    --configs solver/mega_configs/mega_base_mz_8k.json \
    --results_dir results/data/mega
