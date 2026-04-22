#!/bin/bash
#SBATCH --job-name=vlora_p
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_vision_lora_palig_%j.log
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
# r=32 on PaLiGemma2-10B ≈ 80M params (controller was 140M). Closer comparison would use r=64.
$PYTHON solver/eval_vision_lora.py \
    --model models/full/paligemma2-10b \
    --lora_r 32 --total_steps 5000 --seed 42 --grad_accum 16 \
    --results_dir results/data/vision_lora
