#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_dense_long_%j.log
#SBATCH --job-name=den_l

# Dense 172M with 20K steps — fair comparison to ARR-PSRT long run

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Dense 172M Long (20K steps) ===" && echo "Started: $(date)"

envs/deeppass/bin/python psrt/train_dense_baseline.py \
    --size 172m \
    --total_steps 20000 \
    --batch_size 8 \
    --seq_len 512 \
    --lr 3e-4

echo "=== Finished: $(date) ==="
