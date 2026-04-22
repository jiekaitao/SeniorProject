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

# ARR-PSRT 1.1B v16c: Joint training, LR=7.5e-5 (half of v16b)
# v16b (1e-4) shows growing K=2 advantage (-45 at step 8K)
# v16 (1.5e-4) oscillates. Lower LR = more stable advantage.
# v16c tests if even lower LR gives smoother convergence.
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== ARR-PSRT 1.1B v16c (joint, LR=7.5e-5) ===" && echo "Started: $(date)"

envs/deeppass/bin/python psrt/train_arr.py \
    --size 1b \
    --total_steps 100000 \
    --batch_size 4 \
    --seq_len 2048 \
    --lr 7.5e-5 \
    --joint \
    --shared_scale 1.0

echo "=== Finished: $(date) ==="
