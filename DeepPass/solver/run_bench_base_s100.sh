#!/bin/bash
#SBATCH --job-name=bench_b2
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_bench_base_s100_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=14:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python
$PYTHON solver/mega_runner_benchmarks.py \
    --model models/full/Llama-3.1-8B \
    --benchmarks hellaswag,winogrande,boolq \
    --inject_layer 12 \
    --n_rounds 5 \
    --total_steps 8000 \
    --seed 100 \
    --grad_accum 16
