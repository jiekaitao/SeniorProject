#!/bin/bash
#SBATCH --job-name=hybrid_ia
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_hybrid_inst_a_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=14:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python
$PYTHON solver/eval_hybrid_controller.py \
    --model models/full/Llama-3.1-8B-Instruct \
    --benchmarks hellaswag,winogrande,boolq \
    --inject_layer 12 \
    --n_rounds 5 \
    --total_steps 8000 \
    --seed 42 \
    --grad_accum 16 \
    --dora_rank 8 \
    --dora_layers "20-31" \
    --lambda_sparse 0.001 \
    --results_dir results/data/hybrid
