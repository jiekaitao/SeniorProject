#!/bin/bash
#SBATCH --job-name=pk_gem
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_packed_gemma_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== PACKED: Gemma 3 27B on benchmarks (lowrank writer built-in) ==="
$PYTHON solver/eval_deliberation_multi_benchmark.py \
    --benchmarks winogrande,piqa,openbookqa,boolq \
    --model models/full/gemma-3-27b-it \
    --rounds 2 --seeds 42 --steps 1500 --grad_accum 8
