#!/bin/bash
#SBATCH --job-name=attn_grid
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_attn_only_grid_%j.log
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

echo "=== Attention-only + FFN-only on spatialgrid ==="
$PYTHON solver/eval_deliberation_attn_ffn.py \
    --modes full,attention_only,ffn_only \
    --task spatialgrid --rounds 3 --seeds 42,7 --steps 2000

echo "=== Attention-only on HellaSwag ==="
$PYTHON solver/eval_deliberation_hellaswag.py \
    --rounds 2,3 --slots 8 --seeds 42 --steps 2000
