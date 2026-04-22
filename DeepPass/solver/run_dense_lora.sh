#!/bin/bash
#SBATCH --job-name=dense_lr
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_dense_lora_%j.log
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

# Additional LoRA baselines in parallel — different seeds for robustness
$PYTHON solver/mega_runner_lora.py \
    --model models/full/Llama-3.1-8B-Instruct \
    --benchmarks hellaswag --lora_r 64 \
    --total_steps 8000 --seed 100 --grad_accum 16 > /tmp/dense_lora_1.log 2>&1 &
P1=$!

$PYTHON solver/mega_runner_lora.py \
    --model models/full/Llama-3.1-8B-Instruct \
    --benchmarks winogrande --lora_r 64 \
    --total_steps 8000 --seed 100 --grad_accum 16 > /tmp/dense_lora_2.log 2>&1 &
P2=$!

$PYTHON solver/mega_runner_lora.py \
    --model models/full/Llama-3.1-8B-Instruct \
    --benchmarks boolq --lora_r 64 \
    --total_steps 8000 --seed 100 --grad_accum 16 > /tmp/dense_lora_3.log 2>&1 &
P3=$!

echo "Launched 3 LoRA processes: $P1 $P2 $P3"
wait $P1 $P2 $P3
echo "All 3 done"
cat /tmp/dense_lora_*.log | tail -80
