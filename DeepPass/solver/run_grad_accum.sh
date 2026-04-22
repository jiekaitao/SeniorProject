#!/bin/bash
#SBATCH --job-name=ga16
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_grad_accum_%j.log
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

echo "=== Grad Accumulation 16 (effective batch 16) ==="
echo "Comparing GA=1 vs GA=8 vs GA=16 on mazenav"
for GA in 1 8 16; do
    echo "--- grad_accum=$GA ---"
    $PYTHON solver/eval_deliberation_attn_ffn.py \
        --modes full --task mazenav --rounds 3 \
        --seeds 42,7,137 --steps 3000 --grad_accum $GA
done
