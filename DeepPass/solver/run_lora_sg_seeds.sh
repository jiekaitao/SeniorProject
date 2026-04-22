#!/bin/bash
#SBATCH --job-name=lora_sg
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lora_sg_seeds_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=14:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python
# Replicate SpatialGrid LoRA with different seeds to confirm
for S in 100 200 300; do
    $PYTHON solver/mega_runner_lora_spatial.py \
        --model models/full/Llama-3.1-8B \
        --tasks spatialgrid \
        --lora_r 64 --total_steps 8000 --seed $S --grad_accum 16
    $PYTHON solver/mega_runner_lora_spatial.py \
        --model models/full/Llama-3.1-8B-Instruct \
        --tasks spatialgrid \
        --lora_r 64 --total_steps 8000 --seed $S --grad_accum 16
done
