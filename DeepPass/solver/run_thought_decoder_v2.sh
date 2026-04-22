#!/bin/bash
#SBATCH --job-name=thought2
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_thought_decoder_v2_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Thought decoder with 5 rounds (to see if later rounds differ) ==="
$PYTHON solver/eval_thought_decoder.py --seed 7 --n_rounds 5 --steps 3000 --decode_samples 15
