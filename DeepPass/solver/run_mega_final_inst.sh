#!/bin/bash
#SBATCH --job-name=fin_ins
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_mega_final_inst_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python
$PYTHON solver/mega_runner.py \
    --model models/full/Llama-3.1-8B-Instruct \
    --configs solver/mega_configs/mega_final_inst.json \
    --results_dir results/data/mega
