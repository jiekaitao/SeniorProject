#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_psrt_172m_%j.log
#SBATCH --job-name=psrt172

# PSRT-172M: Projected Split-State Recurrent Transformer
# Novel architecture: memory frozen, reasoning iterates
# 3-phase curriculum: K=1 -> K={1,2,3} -> adaptive halting
# ~20K steps, ~164M tokens, ~2-3 hours on B200

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== PSRT-172M Training ===" && echo "Started: $(date)"

envs/deeppass/bin/python psrt/train.py \
    --size 172m \
    --total_steps 20000 \
    --batch_size 8 \
    --seq_len 1024 \
    --lr 3e-4

echo "=== Finished: $(date) ==="
