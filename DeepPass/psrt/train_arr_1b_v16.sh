#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_arr_1b_%j.log
#SBATCH --job-name=arr1b

# ARR-PSRT 1.1B v16: Joint training from step 0 — NO Phase A/B split
# v15/v15b/v15c all show Phase B instability: backbone unfreezing disrupts experts
# New approach: train ALL params together from the start with K=2, uniform routing
# Then gradually introduce routing and K=3 after step 25K
# All scratchpad stability fixes from v14 included
# Backbone LR = full LR (same as experts) — no scaling needed when training jointly
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== ARR-PSRT 1.1B v16 (joint training, no phase split) ===" && echo "Started: $(date)"

envs/deeppass/bin/python psrt/train_arr.py \
    --size 1b \
    --total_steps 100000 \
    --batch_size 4 \
    --seq_len 2048 \
    --lr 1.5e-4 \
    --joint \
    --shared_scale 1.0

echo "=== Finished: $(date) ==="
