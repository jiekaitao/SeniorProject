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

# ARR-PSRT 1.1B v14: GPT-5.4 Pro diagnosis fixes
# Root cause: forward-pass overflow in K=3 recurrent pass
# - Scratchpad was unbounded additive integrator → overflow → NaN cross-attn
# - Zero beta at t=2 was 0.0*inf=NaN tripwire
# Fixes:
#   1. Scratchpad: RMSNorm + decay(0.95) + bounded writes(tanh clip) + output norm
#   2. Skip zero-beta experts (no FFN eval when beta=0.0)
#   3. Phase B1 (first 2K steps): K=2 only. B2: gradually ramp K=3
#   4. FP32 routing losses with clamps
#   5. NaN-TRACE instrumentation to find first bad tensor
#   6. error_if_nonfinite grad clipping
#   7. Safe h_norm (mean-abs instead of norm, clamped)
# Backbone: frozen (isolate forward-pass fixes)
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== ARR-PSRT 1.1B v14 (scratchpad + beta + B1/B2 fix) ===" && echo "Started: $(date)"

envs/deeppass/bin/python psrt/train_arr.py \
    --size 1b \
    --total_steps 100000 \
    --batch_size 4 \
    --seq_len 2048 \
    --lr 1.5e-4 \
    --freeze_shared \
    --resume psrt/checkpoints/arr_psrt/phase_a_end.pt

echo "=== Finished: $(date) ==="
