#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_arr_1b_%j.log
#SBATCH --job-name=arr1b

# ARR-PSRT 1.1B v17: GPT-5.4 Pro architectural redesign
# Root cause: pass 2 had 50% leverage but only 32 dimensions of new info
# Fixes:
#   1. Separate self-attn(r) from cross-attn(r→m₀) — clean split state
#   2. Bank size 16→64 — enough rank for useful reread
#   3. Slot-attention scratch writer — preserves token structure, not rank-1
#   4. Per-pass eta gates (η₁≈0.8, η₂≈0.1) — pass 2 is small correction
#   5. Gated combine — prevents "ignore r" degeneracy
# Joint training from step 0 (proven 8x better than Phase A/B)
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== ARR-PSRT v17 (GPT-5.4 Pro redesign) ===" && echo "Started: $(date)"

envs/deeppass/bin/python psrt/train_arr.py \
    --size 1b \
    --total_steps 100000 \
    --batch_size 4 \
    --seq_len 2048 \
    --lr 1.5e-4 \
    --joint \
    --shared_scale 1.0

echo "=== Finished: $(date) ==="
