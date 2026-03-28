#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_llama3_70b_%j.log
#SBATCH --job-name=ll3_70b

# LLaMA 3 70B — cross-architecture validation
# Uses pre-downloaded model at /data/ai/models/nlp/llama/models_llama3/
# 1. Baseline dual probe
# 2. Sweep single blocks (every 5th layer) to find best
# 3. Test best pair
# 4. Test attn-only vs full on best block

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== LLaMA 3 70B Cross-Architecture Validation ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np

sys.path.insert(0, 'scripts')
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe, _load_questions

MODEL_PATH = '/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-70B-Instruct-hf'
SAVE_DIR = 'results/data/llama3_70b'
os.makedirs(SAVE_DIR, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

print('Loading LLaMA 3 70B...', flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True
)
print(f'Loaded in {time.time()-t0:.0f}s', flush=True)

# Find layers
inner = model.model if hasattr(model, 'model') else model.transformer
layers_attr = 'layers' if hasattr(inner, 'layers') else 'h'
original_layers = list(getattr(inner, layers_attr))
N = len(original_layers)
print(f'Architecture: {type(model).__name__}, {N} layers', flush=True)

eq_all = _load_questions()

def set_num_layers(n):
    if hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = n

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def apply_blocks(blocks):
    order = build_order(blocks, N)
    setattr(inner, layers_attr, nn.ModuleList([original_layers[idx] for idx in order]))
    set_num_layers(len(order))

def restore():
    setattr(inner, layers_attr, nn.ModuleList(original_layers))
    set_num_layers(N)

def generate_no_cache(prompt, max_new_tokens=64):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

def gen(p): return generate_no_cache(p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(p, max_new_tokens=128)

def evaluate(blocks, name, full_eq=False):
    if blocks:
        apply_blocks(blocks)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_all if full_eq else None, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    if blocks:
        restore()
    n_extra = sum(j - i for i, j in blocks) if blocks else 0
    print(f'  {name:45s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} +{n_extra}layers ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'blocks': [list(b) for b in blocks] if blocks else [],
            'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

def save_checkpoint(data, name):
    with open(f'{SAVE_DIR}/{name}.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  [SAVED] {name}.json', flush=True)

# ======================================================================
# 1. Baseline
# ======================================================================
print('\\n=== Baseline ===', flush=True)
baseline = evaluate([], 'baseline', full_eq=True)
save_checkpoint(baseline, 'baseline')

# ======================================================================
# 2. Single block sweep (every 5th layer, sizes 1 and 2)
# ======================================================================
print('\\n=== Single Block Sweep ===', flush=True)
singles = []
for size in [1, 2]:
    for start in range(0, N - size, 5):
        block = [(start, start + size)]
        r = evaluate(block, f'single ({start},{start+size})')
        singles.append(r)

singles.sort(key=lambda x: x['combined'], reverse=True)
print(f'\\nTop 10 singles:', flush=True)
for r in singles[:10]:
    print(f'  {r[\"name\"]:45s} combined={r[\"combined\"]:.2f}', flush=True)
save_checkpoint({'baseline': baseline, 'singles': singles}, 'single_sweep')

# ======================================================================
# 3. Greedy pair from best single
# ======================================================================
print('\\n=== Greedy Pair Search ===', flush=True)
best_single = singles[0]
anchor = [tuple(b) for b in best_single['blocks']]
print(f'Anchor: {anchor}', flush=True)

pairs = []
for size in [1, 2]:
    for start in range(0, N - size, 5):
        cand = (start, start + size)
        # Check no overlap with anchor
        overlaps = any(not (cand[1] <= a[0] or a[1] <= cand[0]) for a in anchor)
        if overlaps:
            continue
        blocks = list(anchor) + [cand]
        r = evaluate(blocks, f'pair {anchor[0]}+({start},{start+size})')
        pairs.append(r)

pairs.sort(key=lambda x: x['combined'], reverse=True)
print(f'\\nTop 5 pairs:', flush=True)
for r in pairs[:5]:
    print(f'  {r[\"name\"]:45s} combined={r[\"combined\"]:.2f}', flush=True)
save_checkpoint({'anchor': best_single, 'pairs': pairs}, 'pair_sweep')

# ======================================================================
# 4. Attn-only vs full on best block (FFN hypothesis test)
# ======================================================================
print('\\n=== Attention-Only vs Full Duplication ===', flush=True)
best_block_spec = [tuple(b) for b in singles[0]['blocks']]
dup_layers = []
for (i, j) in best_block_spec:
    for l in range(i, j):
        dup_layers.append(l)

# Full dup (already have from sweep)
full_r = singles[0]

# Attn-only: zero FFN on second pass
apply_blocks(best_block_spec)
hooks = []
for layer_idx in dup_layers:
    module = original_layers[layer_idx]
    counter = [0]
    def make_zero_ffn(ctr):
        def hook(module, input, output):
            ctr[0] += 1
            if ctr[0] % 2 == 0:
                if isinstance(output, tuple):
                    return (0.0 * output[0],) + output[1:]
                return 0.0 * output
            return output
        return hook
    h = module.mlp.register_forward_hook(make_zero_ffn(counter))
    hooks.append(h)

t0 = time.time()
math_r = run_math_probe(gen, verbose=False)
eq_r = run_eq_bench_probe(gen_long, verbose=False)
attn_combined = math_r['score'] * 50 + eq_r['score'] * 0.5
for h in hooks:
    h.remove()
restore()

attn_r = {'name': f'attn_only {best_block_spec}', 'math': math_r['score'],
           'eq': eq_r['score'], 'combined': attn_combined}
print(f'  attn_only: combined={attn_combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
print(f'  full_dup:  combined={full_r[\"combined\"]:.2f}', flush=True)
print(f'  FFN impact: {attn_combined - full_r[\"combined\"]:+.2f}', flush=True)

# Whisper FFN test
apply_blocks(best_block_spec)
hooks = []
for layer_idx in dup_layers:
    module = original_layers[layer_idx]
    counter = [0]
    def make_whisper(ctr, beta=0.2):
        def hook(module, input, output):
            ctr[0] += 1
            if ctr[0] % 2 == 0:
                if isinstance(output, tuple):
                    return (beta * output[0],) + output[1:]
                return beta * output
            return output
        return hook
    h = module.mlp.register_forward_hook(make_whisper(counter))
    hooks.append(h)

t0 = time.time()
math_r = run_math_probe(gen, verbose=False)
eq_r = run_eq_bench_probe(gen_long, verbose=False)
whisper_combined = math_r['score'] * 50 + eq_r['score'] * 0.5
for h in hooks:
    h.remove()
restore()

whisper_r = {'name': f'whisper_ffn02 {best_block_spec}', 'math': math_r['score'],
              'eq': eq_r['score'], 'combined': whisper_combined}
print(f'  whisper02: combined={whisper_combined:.2f} ({time.time()-t0:.0f}s)', flush=True)

save_checkpoint({
    'best_block': best_block_spec,
    'full_dup': full_r,
    'attn_only': attn_r,
    'whisper_ffn02': whisper_r,
    'ffn_impact_full': attn_combined - full_r['combined'],
    'ffn_impact_whisper': whisper_combined - full_r['combined'],
}, 'sublayer_analysis')

# ======================================================================
# 5. Gate margin measurement
# ======================================================================
print('\\n=== Gate Margin Measurement ===', flush=True)
test_prompts = [
    'What is 127 * 348?', 'What is the capital of France?',
    'Explain entropy simply.', 'What is 2^16?',
    'Who wrote Romeo and Juliet?', 'What is the square root of 152399025?',
]

gate_stats = {}
for layer_idx in dup_layers:
    apply_blocks(best_block_spec)
    module = original_layers[layer_idx]
    gate_proj = module.mlp.gate_proj if hasattr(module.mlp, 'gate_proj') else None

    if gate_proj is None:
        print(f'  Layer {layer_idx}: no gate_proj found, skipping', flush=True)
        restore()
        continue

    flip_rates = []
    for prompt in test_prompts:
        gate_vals = []
        counter = [0]
        def make_capture(ctr, gvals, gp):
            def hook(module, input, output):
                ctr[0] += 1
                with torch.no_grad():
                    inp = input[0] if isinstance(input, tuple) else input
                    g = gp(inp[:, -1, :]).cpu().float()
                    gvals.append(g.squeeze(0))
                return output
            return hook
        h = module.mlp.register_forward_hook(make_capture(counter, gate_vals, gate_proj))
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).to(next(model.parameters()).device)
        with torch.no_grad():
            model(inputs['input_ids'], use_cache=False)
        h.remove()

        if len(gate_vals) >= 2:
            flips = ((gate_vals[0] > 0) != (gate_vals[1] > 0)).float().mean().item()
            flip_rates.append(flips)

    restore()
    avg_flip = float(np.mean(flip_rates)) if flip_rates else -1
    gate_stats[str(layer_idx)] = {'avg_flip_rate': avg_flip}
    stability = 'stable' if avg_flip < 0.05 else ('moderate' if avg_flip < 0.2 else 'unstable')
    print(f'  Layer {layer_idx}: flip_rate={avg_flip:.4f} ({stability})', flush=True)

save_checkpoint(gate_stats, 'gate_margins')

print(f'\\n{\"=\" * 60}', flush=True)
print('COMPLETE', flush=True)
print(f'{\"=\" * 60}', flush=True)
"

echo "=== Finished: $(date) ==="
