#!/bin/bash
#SBATCH --job-name=d_kscale
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_kscaling_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00

module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6

cd /blue/cis4914/jietao/DeepPass

echo "=== K-Scaling Experiment ==="
echo "Train 3 rounds -> eval at 1,2,3,5,8,10,15,20 rounds"

PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python
$PYTHON solver/eval_deliberation_kscaling.py \
    --train_rounds 3 --slots 8 --seed 42 --steps 3000 \
    --eval_rounds 1,2,3,5,8,10,15,20

echo "=== K-Scaling seed 7 ==="
$PYTHON solver/eval_deliberation_kscaling.py \
    --train_rounds 3 --slots 8 --seed 7 --steps 3000 \
    --eval_rounds 1,2,3,5,8,10,15,20
