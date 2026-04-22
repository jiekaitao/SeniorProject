#!/bin/bash
#SBATCH --job-name=dense_ia
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_dense_inst_s42_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=2
#SBATCH --time=14:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

# 4 benchmarks in parallel on same GPU (shares 192GB VRAM)
# Each process ~25GB VRAM, total ~100GB → leaves headroom
$PYTHON solver/mega_runner_benchmarks.py \
    --model models/full/Llama-3.1-8B-Instruct \
    --benchmarks hellaswag --inject_layer 12 --n_rounds 5 \
    --total_steps 8000 --seed 200 --grad_accum 16 > /tmp/dense_ia_1.log 2>&1 &
P1=$!

$PYTHON solver/mega_runner_benchmarks.py \
    --model models/full/Llama-3.1-8B-Instruct \
    --benchmarks winogrande --inject_layer 12 --n_rounds 5 \
    --total_steps 8000 --seed 200 --grad_accum 16 > /tmp/dense_ia_2.log 2>&1 &
P2=$!

$PYTHON solver/mega_runner_benchmarks.py \
    --model models/full/Llama-3.1-8B-Instruct \
    --benchmarks boolq --inject_layer 12 --n_rounds 5 \
    --total_steps 8000 --seed 200 --grad_accum 16 > /tmp/dense_ia_3.log 2>&1 &
P3=$!

$PYTHON solver/mega_runner_benchmarks.py \
    --model models/full/Llama-3.1-8B-Instruct \
    --benchmarks commonsenseqa --inject_layer 12 --n_rounds 5 \
    --total_steps 8000 --seed 200 --grad_accum 16 > /tmp/dense_ia_4.log 2>&1 &
P4=$!

echo "Launched 4 processes: $P1 $P2 $P3 $P4"
wait $P1 $P2 $P3 $P4
echo "All 4 processes done"
cat /tmp/dense_ia_*.log | tail -100
