#!/bin/bash
# Final experiments: Junction FT (all configs) then lm-eval (full benchmarks)
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass

S="/blue/cis4914/jietao/DeepPass/scripts"
R="/blue/cis4914/jietao/DeepPass/results"
M="/blue/cis4914/jietao/DeepPass/models"
LOG="$R/final_experiments_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "FINAL EXPERIMENTS — $(date)"
echo "=========================================="

# Phase 1: Comprehensive Junction FT (~1.5 hours)
echo ""
echo "=== PHASE 1: Junction Fine-Tuning ==="
python -u "$S/comprehensive_junction_ft.py" 2>&1

# Phase 2: lm-eval with our best config (50,60)
# First need to save the (50,60) duplicated model
echo ""
echo "=== PHASE 2: Save (50,60) duplicated model ==="
python -u -c "
import sys, os, copy, torch
import torch.nn as nn
sys.path.insert(0, '$S')
from layer_duplicator import load_original_model

MODEL = '$M/full/calme-2.1-qwen2-72b'
SAVE = '$M/full/calme-2.1-qwen2-72b-dup-50-60'

if os.path.exists(SAVE + '/config.json'):
    print('Model already saved, skipping...')
else:
    print('Loading...')
    model, tok = load_original_model(MODEL)
    layers = list(model.model.layers)
    N = len(layers)
    new_layers = layers[:60]
    for idx in range(50, 60):
        new_layers.append(copy.deepcopy(layers[idx]))
    new_layers.extend(layers[60:])
    model.model.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    print(f'Saving {len(new_layers)} layers...')
    os.makedirs(SAVE, exist_ok=True)
    model.save_pretrained(SAVE, max_shard_size='5GB')
    tok.save_pretrained(SAVE)
    print('Done!')
    del model; torch.cuda.empty_cache()
" 2>&1

# Phase 3: lm-eval baseline
TASKS="leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro"

echo ""
echo "=== PHASE 3: lm-eval BASELINE — $(date) ==="
python -m lm_eval \
    --model hf \
    --model_args "pretrained=$M/full/calme-2.1-qwen2-72b,dtype=bfloat16,trust_remote_code=True" \
    --tasks $TASKS \
    --batch_size auto \
    --output_path "$R/lm_eval_baseline_72b" \
    --device cuda 2>&1
echo "=== BASELINE DONE — $(date) ==="

# Phase 4: lm-eval with our best (50,60)
DUP50="$M/full/calme-2.1-qwen2-72b-dup-50-60"
if [ -f "$DUP50/config.json" ]; then
    echo ""
    echo "=== PHASE 4: lm-eval (50,60) — $(date) ==="
    python -m lm_eval \
        --model hf \
        --model_args "pretrained=$DUP50,dtype=bfloat16,trust_remote_code=True" \
        --tasks $TASKS \
        --batch_size auto \
        --output_path "$R/lm_eval_dup50_60_72b" \
        --device cuda 2>&1
    echo "=== (50,60) DONE — $(date) ==="
fi

# Final compilation
python "$S/compile_results.py" 2>&1

echo ""
echo "=========================================="
echo "ALL FINAL EXPERIMENTS COMPLETE — $(date)"
echo "=========================================="
