#!/bin/bash
#SBATCH --job-name=best_comb
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_combined_best_%j.log
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

echo "=== Combined best: lowrank writer + GA16 on mazenav ==="
echo "Testing if lowrank_only + GA16 + train-5 pushes past 72%"

for SEED in 42 7 137; do
    echo "--- seed $SEED ---"
    $PYTHON solver/eval_deliberation_hybrid_writer.py \
        --modes lowrank_only --task mazenav --rounds 5 \
        --seeds $SEED --steps 3000
done
