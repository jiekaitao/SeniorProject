#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_trm_ll3_%j.log
#SBATCH --job-name=trm_ll3

# LLM-to-TRM: Convert LLaMA 3 8B into a Thinking Recursive Model
# Surgically add memory/reasoning projections around core [10,13)
# Only ~8M new params trained — everything else frozen
# Tests whether the PSRT split-state design works on pre-trained models

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
export CUDA_VISIBLE_DEVICES=0
echo "=== LLM-to-TRM: LLaMA 3 8B ===" && echo "Started: $(date)"

envs/deeppass/bin/python sirt/llm_to_trm.py \
    --model /data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf \
    --name llama3_8b \
    --core_start 10 --core_end 13 \
    --max_steps 500 --lr 1e-4

echo "=== Finished: $(date) ==="
