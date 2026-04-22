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

# ARR-PSRT 1.1B v13: Router zero-init + stronger entropy (0.5)
# v12 proved backbone freeze doesn't help — NaN from router one-hot collapse
# Fixes: (1) zero-init router at Phase B start (was random, never trained in Phase A)
#        (2) entropy coeff 0.1→0.5 (was only 1.4% of loss, now 7%)
# Also keeping backbone frozen to isolate router fix effect
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== ARR-PSRT 1.1B v13 (zero-init router + 0.5 entropy) ===" && echo "Started: $(date)"

envs/deeppass/bin/python psrt/train_arr.py \
    --size 1b \
    --total_steps 100000 \
    --batch_size 4 \
    --seq_len 2048 \
    --lr 1.5e-4 \
    --freeze_shared \
    --resume psrt/checkpoints/arr_psrt/phase_a_end.pt

echo "=== Finished: $(date) ==="
