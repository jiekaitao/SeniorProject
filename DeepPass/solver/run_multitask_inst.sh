#!/bin/bash
#SBATCH --job-name=mt_inst
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_multitask_inst_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python
# Train on 4 benchmarks, evaluate on 4 held-out. Direct test of zero-shot transfer.
$PYTHON solver/mega_runner_multitask.py \
    --model models/full/Llama-3.1-8B-Instruct \
    --train_benchmarks hellaswag,winogrande,boolq,spatialgrid \
    --heldout_benchmarks openbookqa,commonsenseqa,spatialmap,mazenav \
    --inject_layer 12 --n_rounds 5 --total_steps 16000 \
    --seed 42 --grad_accum 16 --results_dir results/data/multitask
