#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_dense_172m_%j.log
#SBATCH --job-name=den172

# Dense 172M baseline — same data, same params, no recursion
# For fair comparison against ARR-PSRT 172M

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Dense 172M Baseline ===" && echo "Started: $(date)"

envs/deeppass/bin/python psrt/train_dense_baseline.py \
    --size 172m \
    --total_steps 12000 \
    --batch_size 8 \
    --seq_len 512 \
    --lr 3e-4

echo "=== Finished: $(date) ==="
