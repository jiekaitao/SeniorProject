#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ft_llama3_v4_%j.log
#SBATCH --job-name=ft_ll3v4

# LLaMA 3 8B v4: Mix K=1 and K=2 training to preserve K=1
# v3 got K=2=+3.22 but K=1=-2.60
# Fix: alternate between K=1 (preserve) and K=2 (learn recursion) every step

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== LLaMA 3 8B Recursion FT v4 (mixed K) ===" && echo "Started: $(date)"

envs/deeppass/bin/python -c "
import sys
sys.path.insert(0, 'sirt')
sys.path.insert(0, 'scripts')
from recursion_finetune import *

# Override the curriculum weights to heavily favor K=1
import recursion_finetune as rf

# Monkey-patch: 70% K=1, 25% K=2, 5% K=3
original_ft = rf.finetune_recursion
def patched_ft(model, tokenizer, device, data_dir, core_start, core_end, **kwargs):
    # Patch random.choices inside the function
    import random as _r
    _orig_choices = _r.choices
    def _patched_choices(population, weights=None, k=1):
        return _orig_choices([1, 2, 3], weights=[0.7, 0.25, 0.05], k=k)
    _r.choices = _patched_choices
    try:
        return original_ft(model, tokenizer, device, data_dir, core_start, core_end, **kwargs)
    finally:
        _r.choices = _orig_choices

rf.finetune_recursion = patched_ft

# Run with same params as v3
main_args = [
    '--model', '/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf',
    '--name', 'llama3_8b_v4',
    '--core_start', '10', '--core_end', '13',
    '--max_steps', '300', '--lr', '5e-7',
    '--batch_size', '1', '--seq_len', '1024',
]
sys.argv = ['recursion_finetune.py'] + main_args

import os
os.makedirs('sirt/recursion_ft/llama3_8b_v4', exist_ok=True)
main()
"

echo "=== Finished: $(date) ==="
