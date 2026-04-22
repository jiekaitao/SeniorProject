#!/bin/bash
#SBATCH --job-name=distl_wg
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_distill_wg_%j.log
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
# Key experiment: if LoRA teacher can teach controller semantic associations
$PYTHON solver/mega_runner_distill.py \
    --model models/full/Llama-3.1-8B-Instruct \
    --benchmark winogrande \
    --teacher_steps 6000 --student_steps 8000 \
    --n_cache 4000 --kl_weight 0.7 --seed 42 --grad_accum 16
