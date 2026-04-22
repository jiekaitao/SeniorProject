#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_dar_1b_%j.log
#SBATCH --job-name=dar1b

# DAR 1B: Dense Attention Replay
# Standard 24-layer transformer + attention replay on middle 6 layers
# Dense containment: gates=0 → exact standard transformer
# Near-zero param tax: ~25K replay params out of ~1.7B total
# Mixed K training: 50% K=1, 50% K=2 with explicit K=1 aux loss
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== DAR 1B ===" && echo "Started: $(date)"

envs/deeppass/bin/python psrt/train_dar.py \
    --size 1b \
    --total_steps 100000 \
    --batch_size 4 \
    --seq_len 2048 \
    --lr 1.5e-4

echo "=== Finished: $(date) ==="
