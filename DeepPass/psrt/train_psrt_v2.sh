#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_psrt_v2_%j.log
#SBATCH --job-name=psrt_v2

# PSRT v2: Mixed data training
# Fix from v1: fineweb-edu alone too easy, model learned E[K]=1.0
# Now: 50% general + 25% math + 25% science
# Lower halt penalty (0.001 vs 0.01) to let model explore recursion
# Heavier K=2,3 curriculum in Phase 2

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== PSRT v2: Mixed Data Training ===" && echo "Started: $(date)"

envs/deeppass/bin/python psrt/train_v2.py \
    --size 172m \
    --total_steps 20000 \
    --batch_size 8 \
    --seq_len 1024 \
    --lr 3e-4 \
    --halt_penalty 0.001

echo "=== Finished: $(date) ==="
