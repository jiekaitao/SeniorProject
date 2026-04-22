#!/bin/bash
#SBATCH --job-name=ga16_bench
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ga16_benchmarks_%j.log
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

echo "=== GA=16 on HellaSwag ==="
$PYTHON solver/eval_deliberation_hellaswag.py \
    --rounds 1,2,3 --slots 8 --seeds 42,7 --steps 3000

echo "=== GA=16 on ARC ==="
$PYTHON solver/eval_deliberation_arc.py \
    --rounds 1,2,3 --slots 8 --seeds 42,7 --steps 2000
