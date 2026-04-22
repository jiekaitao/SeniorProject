#!/bin/bash
#SBATCH --job-name=inst_lrnk
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_llama_instruct_lowrank_%j.log
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
echo "=== Llama 3.1 8B Instruct with lowrank writer ==="
$PYTHON solver/eval_deliberation_multi_benchmark.py \
    --benchmarks winogrande,piqa,openbookqa \
    --model models/full/Llama-3.1-8B-Instruct \
    --rounds 3 --seeds 42,7 --steps 2000 --grad_accum 8
