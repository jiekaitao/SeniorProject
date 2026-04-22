#!/bin/bash
#SBATCH --job-name=ga16_xtsk
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ga16_cross_task_%j.log
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

echo "=== GA=16 on all SpatialEval tasks ==="
for TASK in mazenav spatialmap spatialgrid; do
    echo "--- $TASK ---"
    $PYTHON solver/eval_deliberation_attn_ffn.py \
        --modes full --task $TASK --rounds 3 \
        --seeds 42,7,137 --steps 3000 --grad_accum 16
done
