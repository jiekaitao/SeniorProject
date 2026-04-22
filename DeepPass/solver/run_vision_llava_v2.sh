#!/bin/bash
#SBATCH --job-name=vis_ll2
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_vision_llava_v2_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python
$PYTHON solver/eval_deliberation_vision.py \
    --model models/full/llava-v1.6-mistral-7b-hf \
    --benchmark vsr \
    --inject_layer 12 \
    --n_rounds 5 \
    --total_steps 5000 \
    --seed 42 \
    --grad_accum 16 \
    --tag vision_llava7b_vsr_L12_5k_s42_v2
