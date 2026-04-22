#!/bin/bash
#SBATCH --job-name=peft_dwg
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_peft_dora_wg_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python
# DoRA on WinoGrande and HellaSwag - does DoRA also dominate LoRA on semantic tasks?
$PYTHON solver/mega_runner_pefts.py \
    --method dora --model models/full/Llama-3.1-8B-Instruct \
    --tasks winogrande --rank 64 --total_steps 8000 --seed 42 --grad_accum 16 \
    --results_dir results/data/peft_variants
