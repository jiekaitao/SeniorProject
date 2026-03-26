#!/bin/bash
# Run EQ-bench on Ng's exact model with key configs
# This verifies: does dual metric (math + EQ-bench) change the ranking?

module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
cd /blue/cis4914/jietao/DeepPass

echo "=============================================="
echo "EQ-BENCH COMPARISON — Started at $(date)"
echo "=============================================="

# 1. Baseline (no duplication)
echo ""
echo "=== EQ-BENCH: Baseline calme-2.1-qwen2-72b ==="
echo "Started at $(date)"
python -m lm_eval \
    --model hf \
    --model_args "pretrained=models/full/calme-2.1-qwen2-72b,dtype=bfloat16,trust_remote_code=True" \
    --tasks eq_bench \
    --batch_size auto \
    --output_path results/eqbench_baseline \
    --device cuda 2>&1 | tee results/eqbench_baseline.log
echo "Baseline done at $(date)"

# 2. Ng's config (45,52)
echo ""
echo "=== EQ-BENCH: (45,52) Ng's config ==="
echo "Started at $(date)"
python -m lm_eval \
    --model hf \
    --model_args "pretrained=models/full/calme-2.1-qwen2-72b-dup-45-52,dtype=bfloat16,trust_remote_code=True" \
    --tasks eq_bench \
    --batch_size auto \
    --output_path results/eqbench_dup45_52 \
    --device cuda 2>&1 | tee results/eqbench_dup45_52.log
echo "(45,52) done at $(date)"

# 3. Our config (50,60)
echo ""
echo "=== EQ-BENCH: (50,60) Our config ==="
echo "Started at $(date)"
python -m lm_eval \
    --model hf \
    --model_args "pretrained=models/full/calme-2.1-qwen2-72b-dup-50-60,dtype=bfloat16,trust_remote_code=True" \
    --tasks eq_bench \
    --batch_size auto \
    --output_path results/eqbench_dup50_60 \
    --device cuda 2>&1 | tee results/eqbench_dup50_60.log
echo "(50,60) done at $(date)"

echo ""
echo "=============================================="
echo "ALL DONE at $(date)"
echo "=============================================="
