#!/bin/bash
#SBATCH --job-name=sg_ctl_ni
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_surgery_ctrl_nox_inst_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00

module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6

cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Mazenav surgery: controller on Llama-3.1-8B-Instruct, variant=no_x ==="
$PYTHON solver/mega_runner_surgery.py \
    --method controller \
    --model models/full/Llama-3.1-8B-Instruct \
    --variant no_x \
    --inject_layer 12 \
    --n_rounds 5 \
    --total_steps 8000 \
    --seed 42 \
    --grad_accum 16 \
    --results_dir results/data/surgery
