#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_kv_v5_%j.log
#SBATCH --job-name=deeppass_kvv5

# v5: Deep copy + clear layer_types + verify match + compare speed + 1% lm-eval

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== KV Cache v5: Deep copy + layer_types=None ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, copy, gc, time, torch, torch.nn as nn
sys.path.insert(0, 'scripts')
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('Loading on CPU...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map='cpu', dtype=torch.bfloat16, trust_remote_code=True
)

inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)

# Duplicate (45,52) with deep copy
i, j = 45, 52
new_layers = []
for idx in range(N):
    new_layers.append(original_layers[idx])
    if idx == j - 1:
        for dup_idx in range(i, j):
            new_layers.append(copy.deepcopy(original_layers[dup_idx]))

NEW_N = len(new_layers)
inner.layers = nn.ModuleList(new_layers)
model.config.num_hidden_layers = NEW_N
model.config.layer_types = None  # Clear so cache uses num_hidden_layers directly
print(f'Built: {NEW_N} layers (deep copied), layer_types=None', flush=True)

# Set unique layer_idx on each layer (deep copies have their own)
for pos, layer in enumerate(inner.layers):
    if hasattr(layer, 'self_attn'):
        layer.self_attn.layer_idx = pos

# Verify cache
tc = DynamicCache(config=model.config)
print(f'Cache layers: {len(tc.layers)} (expected {NEW_N})', flush=True)
assert len(tc.layers) == NEW_N

# Move to GPU
print('Moving to GPU...', flush=True)
model = model.to('cuda')
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB', flush=True)

# Test
input_ids = tokenizer('What is the capital of France?', return_tensors='pt').to('cuda')
out_cache = model.generate(**input_ids, max_new_tokens=30, use_cache=True)
out_nocache = model.generate(**input_ids, max_new_tokens=30, use_cache=False)
rc = tokenizer.decode(out_cache[0], skip_special_tokens=True)
rn = tokenizer.decode(out_nocache[0], skip_special_tokens=True)
print(f'Cache ON:  {rc[:100]}', flush=True)
print(f'Cache OFF: {rn[:100]}', flush=True)
min_len = min(len(out_cache[0]), len(out_nocache[0]))
match = (out_cache[0][:min_len] == out_nocache[0][:min_len]).all().item()
print(f'MATCH: {match}', flush=True)

# Speed
prompt = 'The theory of general relativity describes gravity as curvature of spacetime.'
inp = tokenizer(prompt, return_tensors='pt').to('cuda')
model.generate(**inp, max_new_tokens=5, use_cache=True)
t0 = time.time()
for _ in range(3): model.generate(**inp, max_new_tokens=64, use_cache=True)
ct = (time.time() - t0) / 3
t0 = time.time()
for _ in range(3): model.generate(**inp, max_new_tokens=64, use_cache=False)
nt = (time.time() - t0) / 3
print(f'Cache: {64/ct:.1f} tok/s  NoCache: {64/nt:.1f} tok/s  Speedup: {nt/ct:.1f}x', flush=True)

if match:
    print(f'\\n--- lm-eval 1% ---', flush=True)
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator
    TASKS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
             'leaderboard_musr', 'leaderboard_mmlu_pro']
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
    r = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.01)
    for task in TASKS:
        data = r['results'].get(task, {})
        for m in sorted(data.keys()):
            if isinstance(data[m], (int, float)) and 'stderr' not in m:
                print(f'  {task}/{m}: {data[m]:.4f}', flush=True)
    os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
    with open('results/data/72b/lm_eval/proper/kv_v5.json', 'w') as f:
        json.dump({'method': 'deep_copy + layer_types=None', 'match': match, 'speedup': nt/ct,
                   'results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in r['results'].items()}}, f, indent=2)
    print('Saved!', flush=True)
else:
    print('MATCH FAILED', flush=True)
"

echo "=== Done at $(date) ==="
