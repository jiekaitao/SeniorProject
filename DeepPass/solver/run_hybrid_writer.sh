#!/bin/bash
#SBATCH --job-name=hyb_writ
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_hybrid_writer_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6

cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Hybrid Thought Writer (vocab + low-rank residual) ==="
echo "GPT-5.4 recommendation: m = E^T alpha + U*beta"
$PYTHON solver/eval_deliberation_hybrid_writer.py \
    --modes vocab_only,hybrid_r32,hybrid_r64,lowrank_only \
    --task mazenav --rounds 3 --seeds 42,7 --steps 3000

echo ""
echo "=== Cross-task: spatialmap ==="
$PYTHON solver/eval_deliberation_hybrid_writer.py \
    --modes vocab_only,hybrid_r64 \
    --task spatialmap --rounds 3 --seeds 42,7 --steps 2000
