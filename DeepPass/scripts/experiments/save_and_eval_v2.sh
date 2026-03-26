#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_save_eval_v2_%j.log
#SBATCH --job-name=deeppass_sev2

# Step 1: Save Ng's (45,52) duplicated model with deep-copied layers
# Step 2: Run lm-eval properly with KV cache (1% test)
# Step 3: Compare with Ng's reported numbers

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Save + Proper Eval v2 ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, copy, gc, torch, torch.nn as nn
sys.path.insert(0, 'scripts')

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
SAVE_DIR = 'models/saved/ng_45_52_deepcopy'

# =====================================================================
# STEP 1: Save with deep-copied layers
# =====================================================================
print('=' * 70)
print('STEP 1: Save duplicated model with deep-copied layers')
print('=' * 70, flush=True)

if os.path.exists(os.path.join(SAVE_DIR, 'config.json')):
    print(f'Model already saved at {SAVE_DIR}, skipping save step.', flush=True)
else:
    print('Loading base model...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map='cpu', dtype=torch.bfloat16, trust_remote_code=True
    )

    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)
    print(f'Loaded on CPU: {N} layers', flush=True)

    i, j = 45, 52
    print(f'Deep copying layers [{i},{j}) ...', flush=True)

    new_layers = []
    for idx in range(N):
        new_layers.append(original_layers[idx])
        if idx == j - 1:
            for dup_idx in range(i, j):
                print(f'  Deep copying layer {dup_idx}...', flush=True)
                new_layers.append(copy.deepcopy(original_layers[dup_idx]))

    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    print(f'New model: {len(new_layers)} layers', flush=True)

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f'Saving to {SAVE_DIR}...', flush=True)
    model.save_pretrained(SAVE_DIR, max_shard_size='10GB')
    tokenizer.save_pretrained(SAVE_DIR)
    print('Saved!', flush=True)

    del model, inner, original_layers, new_layers
    gc.collect()

# =====================================================================
# STEP 2: Load saved model and verify KV cache
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('STEP 2: Load saved model, verify KV cache')
print(f'{\"=\" * 70}', flush=True)

model = AutoModelForCausalLM.from_pretrained(
    SAVE_DIR, device_map='auto', dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR, trust_remote_code=True)
print(f'Loaded saved model: {model.config.num_hidden_layers} layers', flush=True)
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB', flush=True)

# Test KV cache
input_ids = tokenizer('What is 2+2?', return_tensors='pt').to(model.device)
out_cache = model.generate(**input_ids, max_new_tokens=20, use_cache=True)
out_nocache = model.generate(**input_ids, max_new_tokens=20, use_cache=False)
resp_cache = tokenizer.decode(out_cache[0], skip_special_tokens=True)
resp_nocache = tokenizer.decode(out_nocache[0], skip_special_tokens=True)
print(f'  Cache ON:  {resp_cache[:80]}', flush=True)
print(f'  Cache OFF: {resp_nocache[:80]}', flush=True)
match = (out_cache[0][:min(len(out_cache[0]),len(out_nocache[0]))] == out_nocache[0][:min(len(out_cache[0]),len(out_nocache[0]))]).all().item()
print(f'  Match: {match}', flush=True)

# Speed test
import time
prompt = 'The theory of general relativity describes gravity as curvature of spacetime caused by mass and energy.'
inp = tokenizer(prompt, return_tensors='pt').to(model.device)
model.generate(**inp, max_new_tokens=5, use_cache=True)  # warmup

t0 = time.time()
for _ in range(3):
    model.generate(**inp, max_new_tokens=64, use_cache=True)
cache_time = (time.time() - t0) / 3

t0 = time.time()
for _ in range(3):
    model.generate(**inp, max_new_tokens=64, use_cache=False)
nocache_time = (time.time() - t0) / 3

print(f'  Cache ON: {64/cache_time:.1f} tok/s', flush=True)
print(f'  Cache OFF: {64/nocache_time:.1f} tok/s', flush=True)
print(f'  Speedup: {nocache_time/cache_time:.1f}x', flush=True)

# =====================================================================
# STEP 3: lm-eval (1% subsample) — proper setup
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('STEP 3: lm-eval on saved model (1% subsample)')
print(f'{\"=\" * 70}', flush=True)

from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

lm = HFLM(
    pretrained=SAVE_DIR,
    dtype='bfloat16',
    batch_size='auto',
    device_map_option='auto',
    trust_remote_code=True,
)

TASKS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
         'leaderboard_musr', 'leaderboard_mmlu_pro']

results = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.01)

print(f'\\n=== NG (45,52) RESULTS (1% test, KV cache, saved model) ===', flush=True)
for task in sorted(results['results'].keys()):
    data = results['results'][task]
    # Only print main metrics (not subtasks)
    if task in ['leaderboard_bbh', 'leaderboard_math_hard', 'leaderboard_ifeval',
                'leaderboard_musr', 'leaderboard_mmlu_pro']:
        for metric in sorted(data.keys()):
            if isinstance(data[metric], (int, float)) and 'stderr' not in metric:
                print(f'  {task}/{metric}: {data[metric]:.4f}', flush=True)

# Compare with Ng's reported numbers (scaled)
print(f'\\n=== COMPARISON WITH NG\\'S REPORTED NUMBERS ===', flush=True)
ng_reported = {
    'leaderboard_ifeval': 79.96,
    'leaderboard_bbh': 58.77,
    'leaderboard_math_hard': 38.97,
    'leaderboard_musr': 23.72,
    'leaderboard_mmlu_pro': 49.20,
}
print(f'{\"Task\":>25s} {\"Ng reported\":>12s} {\"Our (1%!)\":>12s} {\"Note\":>s}')
for task in TASKS:
    ng_val = ng_reported.get(task, 0)
    # Find our main metric
    our_val = 0
    data = results['results'].get(task, {})
    for metric in ['acc_norm,none', 'exact_match,none', 'inst_level_loose_acc,none', 'acc,none',
                   'inst_level_strict_acc,none', 'prompt_level_strict_acc,none']:
        if metric in data:
            our_val = data[metric]
            # Try scaling to 0-100 if Ng's is in that range
            if ng_val > 1 and our_val <= 1:
                our_val_scaled = our_val * 100
            else:
                our_val_scaled = our_val
            print(f'{task:>25s} {ng_val:12.2f} {our_val_scaled:12.2f} metric={metric}')
            break

os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
with open('results/data/72b/lm_eval/proper/ng_saved_1pct.json', 'w') as f:
    json.dump({
        'method': 'save_pretrained deep copy, KV cache ON, batch_size auto',
        'subsample': 0.01,
        'cache_speedup': nocache_time/cache_time,
        'outputs_match': match,
        'results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))}
                    for k, v in results['results'].items()},
    }, f, indent=2)
print('\\nSaved!', flush=True)
"

echo "=== Done at $(date) ==="
