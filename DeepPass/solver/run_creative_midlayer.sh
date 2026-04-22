#!/bin/bash
#SBATCH --job-name=cr_midl
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_creative_midlayer_%j.log
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

echo "=== Mid-layer injection sweep (L8, L16, L24) ==="
$PYTHON solver/eval_deliberation_creative.py --experiment midlayer --seeds 42 --steps 3000

echo ""
echo "=== Also: Llama Instruct lowrank on SpatialEval ==="
$PYTHON solver/eval_deliberation_hybrid_writer.py \
    --modes lowrank_only --task mazenav --rounds 3 --seeds 42,7 --steps 2000

$PYTHON solver/eval_deliberation_hybrid_writer.py \
    --modes lowrank_only --task spatialmap --rounds 3 --seeds 42 --steps 2000
