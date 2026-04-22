#!/bin/bash
#SBATCH --job-name=lora_vsr
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lora_vsr_%j.log
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
# LoRA on VSR (vision benchmark) with PaLiGemma - does LoRA match our +33pp?
# Note: VLM LoRA is more complex; first test would be to train on text-only
# For now, do more LoRA text seeds to build variance estimates
$PYTHON solver/mega_runner_lora.py \
    --model models/full/Llama-3.1-8B-Instruct \
    --benchmarks hellaswag,winogrande,boolq \
    --lora_r 64 --total_steps 8000 --seed 200 --grad_accum 16
