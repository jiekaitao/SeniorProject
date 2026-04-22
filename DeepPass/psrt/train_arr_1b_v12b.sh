#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_arr_1b_%j.log
#SBATCH --job-name=arr1b

# ARR-PSRT 1.1B v12b: Diagnostic run — 0.001x shared LR WITH gradient monitoring
# Will NaN around step 26-27K but now with per-component gradient norms
# to identify WHICH component explodes first
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== ARR-PSRT 1.1B v12b (diagnostic) ===" && echo "Started: $(date)"

envs/deeppass/bin/python psrt/train_arr.py \
    --size 1b \
    --total_steps 100000 \
    --batch_size 4 \
    --seq_len 2048 \
    --lr 1.5e-4 \
    --shared_scale 0.001 \
    --resume psrt/checkpoints/arr_psrt/phase_a_end.pt

echo "=== Finished: $(date) ==="
