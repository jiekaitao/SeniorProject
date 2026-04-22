#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_arr_172m_%j.log
#SBATCH --job-name=arr172

# ARR-PSRT 172M — quick Phase B stability test
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== ARR-PSRT 172M Phase B Test ===" && echo "Started: $(date)"

envs/deeppass/bin/python psrt/train_arr.py \
    --size 172m \
    --total_steps 12000 \
    --batch_size 8 \
    --seq_len 512 \
    --lr 3e-4

echo "=== Finished: $(date) ==="
