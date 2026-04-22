#!/bin/bash
#SBATCH --job-name=gem_glob
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_gemma_global_taps_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6

cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Gemma 3 Global-Layer Taps (GPT-5.4 fix) ==="
echo "Pattern=6: global layers at 5,11,17,23,29,35,41,47,53,59"
echo "Previous auto-taps [15,30,45] were ALL local layers!"
echo ""

echo "--- Config 1: Global-only taps at 0.25/0.5/0.75/0.9 depth = [17,29,47,59] ---"
$PYTHON solver/eval_deliberation_general.py \
    --model models/full/gemma-3-27b-it \
    --task mazenav --rounds 2 --slots 8 \
    --seeds 42,7 --steps 2000 \
    --tapped 17,29,47,59

echo ""
echo "--- Config 2: Early+mid+late global = [11,29,53] ---"
$PYTHON solver/eval_deliberation_general.py \
    --model models/full/gemma-3-27b-it \
    --task mazenav --rounds 2 --slots 8 \
    --seeds 42,7 --steps 2000 \
    --tapped 11,29,53

echo ""
echo "--- Config 3: Old local taps [15,30,45] as control ---"
$PYTHON solver/eval_deliberation_general.py \
    --model models/full/gemma-3-27b-it \
    --task mazenav --rounds 2 --slots 8 \
    --seeds 42 --steps 2000 \
    --tapped 15,30,45
