#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lora_ll3_%j.log
#SBATCH --job-name=lora_ll3
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== LLaMA 3 8B Pass-2 LoRA ===" && echo "Started: $(date)"
envs/deeppass/bin/python sirt/lora_recursion.py \
    --model /data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf \
    --name llama3_8b --core_start 10 --core_end 13 \
    --rank 8 --max_steps 300 --lr 5e-5 --contrastive
echo "=== Finished: $(date) ==="
