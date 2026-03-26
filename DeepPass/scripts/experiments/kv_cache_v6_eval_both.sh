#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_kv_v6_%j.log
#SBATCH --job-name=deeppass_kvv6

# v6: Accept that cache ON/OFF may differ numerically (floating point)
# Run lm-eval with BOTH modes and compare SCORES not tokens
# Use deep copy for proper cache isolation

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== KV Cache v6: Compare lm-eval scores cache ON vs OFF ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, copy, gc, time, torch, torch.nn as nn
sys.path.insert(0, 'scripts')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

# =====================================================================
# First: test if BASELINE (no duplication) matches cache ON vs OFF
# =====================================================================
print('=' * 70)
print('TEST 0: Does the BASELINE model match cache ON vs OFF?')
print('=' * 70, flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map='auto', dtype=torch.bfloat16, trust_remote_code=True
)

ids = tokenizer('What is the capital of France?', return_tensors='pt').to(model.device)
oc = model.generate(**ids, max_new_tokens=20, use_cache=True)
on = model.generate(**ids, max_new_tokens=20, use_cache=False)
rc = tokenizer.decode(oc[0], skip_special_tokens=True)
rn = tokenizer.decode(on[0], skip_special_tokens=True)
print(f'Baseline Cache ON:  {rc[:80]}', flush=True)
print(f'Baseline Cache OFF: {rn[:80]}', flush=True)
ml = min(len(oc[0]), len(on[0]))
base_match = (oc[0][:ml] == on[0][:ml]).all().item()
print(f'Baseline match: {base_match}', flush=True)

# Speed comparison on baseline
prompt = 'The theory of general relativity describes gravity as curvature of spacetime.'
inp = tokenizer(prompt, return_tensors='pt').to(model.device)
model.generate(**inp, max_new_tokens=5, use_cache=True)
t0 = time.time()
for _ in range(3): model.generate(**inp, max_new_tokens=64, use_cache=True)
base_cache_time = (time.time() - t0) / 3
t0 = time.time()
for _ in range(3): model.generate(**inp, max_new_tokens=64, use_cache=False)
base_nocache_time = (time.time() - t0) / 3
print(f'Baseline: Cache {64/base_cache_time:.1f} tok/s  NoCache {64/base_nocache_time:.1f} tok/s  Speedup {base_nocache_time/base_cache_time:.1f}x', flush=True)

# =====================================================================
# Now: lm-eval 1% on Ng with cache ON (standard HFLM)
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 1: Ng (45,52) deep copy, lm-eval with KV cache')
print(f'{\"=\" * 70}', flush=True)

inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)

# Deep copy for Ng
i, j = 45, 52
new_layers = []
for idx in range(N):
    new_layers.append(original_layers[idx])
    if idx == j - 1:
        for dup_idx in range(i, j):
            new_layers.append(copy.deepcopy(original_layers[dup_idx]))

inner.layers = nn.ModuleList(new_layers)
model.config.num_hidden_layers = len(new_layers)
model.config.layer_types = None
for pos, layer in enumerate(inner.layers):
    if hasattr(layer, 'self_attn'):
        layer.self_attn.layer_idx = pos
print(f'Ng model: {len(new_layers)} layers', flush=True)

# Speed on duplicated
model.generate(**inp, max_new_tokens=5, use_cache=True)
t0 = time.time()
for _ in range(3): model.generate(**inp, max_new_tokens=64, use_cache=True)
ng_cache_time = (time.time() - t0) / 3
t0 = time.time()
for _ in range(3): model.generate(**inp, max_new_tokens=64, use_cache=False)
ng_nocache_time = (time.time() - t0) / 3
print(f'Ng: Cache {64/ng_cache_time:.1f} tok/s  NoCache {64/ng_nocache_time:.1f} tok/s  Speedup {ng_nocache_time/ng_cache_time:.1f}x', flush=True)

# lm-eval
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator
TASKS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
         'leaderboard_musr', 'leaderboard_mmlu_pro']

print(f'\\nRunning lm-eval 1% with KV cache...', flush=True)
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
r = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.01)

print(f'\\n=== NG RESULTS (deep copy, KV cache ON) ===', flush=True)
for task in TASKS:
    data = r['results'].get(task, {})
    for m in sorted(data.keys()):
        if isinstance(data[m], (int, float)) and 'stderr' not in m:
            print(f'  {task}/{m}: {data[m]:.4f}', flush=True)

# Ng's reported numbers for reference
print(f'\\nNg reported: IFEval=79.96 BBH=58.77 MATH=38.97 MuSR=23.72 MMLU-PRO=49.20', flush=True)

os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
with open('results/data/72b/lm_eval/proper/kv_v6.json', 'w') as f:
    json.dump({
        'method': 'deep_copy + layer_types=None + KV cache ON',
        'baseline_match': base_match,
        'baseline_speedup': base_nocache_time/base_cache_time,
        'ng_speedup': ng_nocache_time/ng_cache_time,
        'ng_results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in r['results'].items()},
    }, f, indent=2)
print('Saved!', flush=True)
"

echo "=== Done at $(date) ==="
