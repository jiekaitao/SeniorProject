#!/bin/bash
#SBATCH --job-name=ks_robust
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_kscaling_robust_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6

cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Train-5 seeds 137, 1337, 2024 for statistical robustness ==="
for SEED in 137 1337 2024; do
    echo "--- seed $SEED ---"
    $PYTHON solver/eval_deliberation_kscaling.py \
        --train_rounds 5 --slots 8 --seed $SEED --steps 3000 \
        --eval_rounds 1,2,3,5,8,10
done
