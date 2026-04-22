#!/bin/bash
#SBATCH --job-name=pk_inst
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_packed_instruct_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== PACKED: Llama 8B Instruct - remaining benchmarks ==="
$PYTHON solver/eval_deliberation_multi_benchmark.py \
    --benchmarks piqa,boolq,commonsenseqa \
    --model models/full/Llama-3.1-8B-Instruct \
    --rounds 3 --seeds 42,7 --steps 2000 --grad_accum 8

echo "=== Instruct on SpatialEval tasks ==="
$PYTHON solver/eval_deliberation_hybrid_writer.py \
    --modes lowrank_only --task mazenav --rounds 3 \
    --seeds 42 --steps 2000

$PYTHON solver/eval_deliberation_hybrid_writer.py \
    --modes lowrank_only --task spatialmap --rounds 3 \
    --seeds 42 --steps 2000
