#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sirt_stage3_%j.log
#SBATCH --job-name=sirt_s3

# SIRT-172M Stage 3: Adaptive halting (ACT loss enabled)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== SIRT-172M Stage 3: Adaptive Halting ==="
echo "Started: $(date)"

$PYTHON sirt/train.py \
    --stage 3 \
    --max_steps 10000 \
    --batch_size 2 \
    --seq_len 4096 \
    --lr 5e-5 \
    --warmup_steps 200 \
    --grad_accum 8 \
    --log_interval 50 \
    --save_interval 2000 \
    --save_dir sirt/checkpoints \
    --data_dir sirt/data \
    --resume sirt/checkpoints/stage2_final.pt

echo "=== Finished: $(date) ==="
