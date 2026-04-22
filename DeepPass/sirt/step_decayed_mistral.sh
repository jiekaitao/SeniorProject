#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_stepdecay_mis_%j.log
#SBATCH --job-name=decay_m

# Track A: Step-decayed FFN on Mistral [28,29)
# Tests 6 beta schedules x 3 K values = 18 configs
# Key test: does front-loading FFN then decaying beat constant beta?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Step-Decayed FFN: Mistral 7B ===" && echo "Started: $(date)"

envs/deeppass/bin/python sirt/step_decayed_ffn.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --name mistral_7b \
    --core_start 28 --core_end 29

echo "=== Finished: $(date) ==="
