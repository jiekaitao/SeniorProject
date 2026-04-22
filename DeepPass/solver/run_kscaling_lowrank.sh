#!/bin/bash
#SBATCH --job-name=ks_lrank
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_kscaling_lowrank_%j.log
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

echo "=== K-Scaling with lowrank writer + train-5 ==="
$PYTHON solver/eval_deliberation_hybrid_writer.py \
    --modes lowrank_only --task mazenav --rounds 5 \
    --seeds 42,7 --steps 3000

echo "=== spatialmap with lowrank ==="
$PYTHON solver/eval_deliberation_hybrid_writer.py \
    --modes lowrank_only --task spatialmap --rounds 3 \
    --seeds 42,7 --steps 3000

echo "=== spatialgrid with lowrank ==="
$PYTHON solver/eval_deliberation_hybrid_writer.py \
    --modes lowrank_only --task spatialgrid --rounds 3 \
    --seeds 42,7 --steps 3000
