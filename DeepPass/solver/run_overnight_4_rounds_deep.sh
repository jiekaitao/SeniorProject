#!/bin/bash
#SBATCH --job-name=ov4
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_overnight_4_rounds_deep_%j.log
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
$PYTHON solver/mega_runner.py \
    --model models/full/Llama-3.1-8B-Instruct \
    --configs solver/mega_configs/overnight_4_rounds_deep.json \
    --results_dir results/data/mega
