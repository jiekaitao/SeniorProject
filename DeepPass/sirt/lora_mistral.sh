#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lora_mis_%j.log
#SBATCH --job-name=lora_mis
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Mistral 7B Pass-2 LoRA ===" && echo "Started: $(date)"
envs/deeppass/bin/python sirt/lora_recursion.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --name mistral_7b --core_start 27 --core_end 30 \
    --rank 8 --max_steps 300 --lr 5e-5 --contrastive
echo "=== Finished: $(date) ==="
