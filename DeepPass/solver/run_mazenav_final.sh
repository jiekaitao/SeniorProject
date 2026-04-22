#!/bin/bash
#SBATCH --job-name=mz_final
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_mazenav_final_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

# Run the SIMPLEST best baseline — standard lowrank with 5 seeds for robustness
echo "=== Lowrank writer: 5 seeds for paper table ==="
for SEED in 42 7 137 2024 1; do
    $PYTHON solver/eval_deliberation_hybrid_writer.py \
        --modes lowrank_only --task mazenav --rounds 3 \
        --seeds $SEED --steps 3000
done
