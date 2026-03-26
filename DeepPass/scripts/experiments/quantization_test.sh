#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_quant_test_%j.log
#SBATCH --job-name=deeppass_quant

# Does layer duplication benefit survive 4-bit quantization?
# Test on Gemma3-27B and 72B in NF4

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Quantization Survival Test ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, gc, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def test_model(model_path, model_name, configs, quant_bits=4):
    print(f'\\n{\"=\" * 70}')
    print(f'MODEL: {model_name} ({quant_bits}-bit NF4)')
    print(f'{\"=\" * 70}', flush=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f'Loading {model_name} in {quant_bits}-bit...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map='auto',
        trust_remote_code=True,
    )

    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    attr = 'layers' if hasattr(inner, 'layers') else 'h'
    original_layers = list(getattr(inner, attr))
    N = len(original_layers)
    vram = torch.cuda.memory_allocated() / 1e9
    print(f'Loaded: {N} layers, {vram:.1f} GB VRAM', flush=True)

    def set_num_layers(n):
        if hasattr(model.config, 'text_config'):
            model.config.text_config.num_hidden_layers = n
        elif hasattr(model.config, 'num_hidden_layers'):
            model.config.num_hidden_layers = n

    def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
    def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

    results = []

    # Baseline
    print('  Baseline...', flush=True)
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    baseline = math_r['score'] * 50 + eq_r['score'] * 0.5
    print(f'  baseline: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={baseline:.2f}', flush=True)
    results.append({'name': 'baseline', 'math': math_r['score'], 'eq': eq_r['score'], 'combined': baseline})

    # Test each config
    for name, blocks in configs:
        order = build_order(blocks, N)
        setattr(inner, attr, nn.ModuleList([original_layers[idx] for idx in order]))
        set_num_layers(len(order))

        math_r = run_math_probe(gen, verbose=False)
        eq_r = run_eq_bench_probe(gen_long, verbose=False)
        combined = math_r['score'] * 50 + eq_r['score'] * 0.5
        delta = combined - baseline
        print(f'  {name:40s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} delta={delta:+.2f}', flush=True)
        results.append({'name': name, 'blocks': [list(b) for b in blocks], 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined, 'delta': delta})

        setattr(inner, attr, nn.ModuleList(original_layers))
        set_num_layers(N)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {'model': model_path, 'model_name': model_name, 'bits': quant_bits,
            'vram_gb': vram, 'num_layers': N, 'baseline': baseline, 'results': results}

# =====================================================================
# Test 1: Gemma3 27B at 4-bit
# =====================================================================
gemma_results = test_model(
    'models/full/gemma-3-27b-it', 'Gemma3-27B',
    configs=[
        ('(20,21) best small single', [(20, 21)]),
        ('(6,11) best large single', [(6, 11)]),
        ('(4,5)+(20,21) best small pair', [(4, 5), (20, 21)]),
    ]
)

# =====================================================================
# Test 2: 72B at 4-bit
# =====================================================================
results_72b = test_model(
    'models/full/calme-2.1-qwen2-72b', '72B',
    configs=[
        ('(45,52) Ng single', [(45, 52)]),
        ('(50,60) our best single', [(50, 60)]),
        ('(0,7)+(45,52) our best pair', [(0, 7), (45, 52)]),
    ]
)

# =====================================================================
# Summary
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('QUANTIZATION SURVIVAL SUMMARY')
print(f'{\"=\" * 70}')

all_output = {'date': datetime.now().isoformat(), 'gemma3': gemma_results, 'qwen72b': results_72b}

for model_data in [gemma_results, results_72b]:
    print(f'\\n{model_data[\"model_name\"]} ({model_data[\"bits\"]}-bit, {model_data[\"vram_gb\"]:.1f}GB):')
    print(f'  Baseline: {model_data[\"baseline\"]:.2f}')
    for r in model_data['results'][1:]:
        print(f'  {r[\"name\"]:40s}: combined={r[\"combined\"]:.2f} delta={r[\"delta\"]:+.2f}')
    has_benefit = any(r.get('delta', 0) > 0 for r in model_data['results'][1:])
    print(f'  Duplication survives quantization: {\"YES\" if has_benefit else \"NO\"}')

os.makedirs('results/data/quantization', exist_ok=True)
with open('results/data/quantization/quant_test_results.json', 'w') as f:
    json.dump(all_output, f, indent=2)
print(f'\\nSaved to results/data/quantization/quant_test_results.json', flush=True)
"

echo "=== Done at $(date) ==="
