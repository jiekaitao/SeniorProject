#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_llama3_8b_%j.log
#SBATCH --job-name=ll3_8b

# LLaMA 3 8B — quick scaling validation
# Tests: baseline, SBUID screen, top 10 singles, best pair, attn-only vs full
# Key question: does duplication help LESS on smaller models? (scaling hypothesis)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== LLaMA 3 8B Quick Validation ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from scipy import stats

sys.path.insert(0, 'scripts')
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe, _load_questions
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = '/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf'
SAVE_DIR = 'results/data/llama3_8b'
os.makedirs(SAVE_DIR, exist_ok=True)

print('Loading LLaMA 3 8B...', flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', dtype=torch.bfloat16)
print(f'Loaded in {time.time()-t0:.0f}s', flush=True)

inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'{N} layers on {device}', flush=True)

eq_all = _load_questions()

def set_num_layers(n): model.config.num_hidden_layers = n
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
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))
def restore():
    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

def gen_no_cache(prompt, max_tokens=64):
    inp = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    for _ in range(max_tokens):
        with torch.no_grad():
            out = model(inp, use_cache=False)
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        if nxt.item() == tokenizer.eos_token_id: break
        inp = torch.cat([inp, nxt], dim=-1)
    plen = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(inp[0, plen:], skip_special_tokens=True)

def gen(p): return gen_no_cache(p, 64)
def gen_long(p): return gen_no_cache(p, 128)

def save(data, name):
    with open(f'{SAVE_DIR}/{name}.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  [SAVED] {name}.json', flush=True)

cal_prompts = [
    'What is 127 * 348?', 'What is 99999 * 99999?',
    'Calculate 15! / 13!', 'What is 2^16?',
    'What is the capital of France?', 'Explain entropy simply.',
]

# ======================================================================
# Baseline
# ======================================================================
print('\\n=== Baseline ===', flush=True)
t0 = time.time()
math_r = run_math_probe(gen, verbose=False)
eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
baseline = math_r['score'] * 50 + eq_r['score'] * 0.5
print(f'Baseline: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={baseline:.2f} ({time.time()-t0:.0f}s)', flush=True)
save({'math': math_r['score'], 'eq': eq_r['score'], 'combined': baseline}, 'baseline')

# ======================================================================
# SBUID screen all single blocks
# ======================================================================
print('\\n=== SBUID Spectral Screen ===', flush=True)
sbuid_data = []
for start in range(N - 1):
    block = (start, start + 1)
    rhos, bloods = [], []
    for prompt in cal_prompts[:4]:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out_base = model(inputs['input_ids'], use_cache=False)
            logits_base = out_base.logits[:, -1, :].float()
            hb, ha, ctr = [None], [None], [0]
            def make_bh(c, b, a):
                def hook(mod, inp, out):
                    c[0] += 1
                    h = out[0] if isinstance(out, tuple) else out
                    if c[0] % 2 == 1: b[0] = h.detach().float()
                    else: a[0] = h.detach().float()
                    return out
                return hook
            bh = original_layers[start].register_forward_hook(make_bh(ctr, hb, ha))
            order = build_order([block], N)
            inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
            set_num_layers(len(order))
            out_dup = model(inputs['input_ids'], use_cache=False)
            logits_dup = out_dup.logits[:, -1, :].float()
            bh.remove()
            restore()
            num = torch.norm(logits_dup - logits_base).item()
            den = torch.norm(logits_base).item()
            if den > 1e-8: rhos.append(num / den)
            if hb[0] is not None and ha[0] is not None:
                bloods.append(torch.norm(ha[0] - hb[0]).item())

    mr = float(np.mean(rhos)) if rhos else 1.0
    mb = float(np.mean(bloods)) if bloods else 0.0
    sbuid_data.append({'block': [start, start+1], 'rho': mr, 'blood': mb,
                       'sbuid_6k': mb - 6000*mr, 'sbuid_10k': mb - 10000*mr})
    if start % 8 == 0:
        print(f'  L{start}: rho={mr:.4f} blood={mb:.1f} sbuid={mb-6000*mr:.0f}', flush=True)

print(f'Screened {len(sbuid_data)} blocks', flush=True)
save(sbuid_data, 'spectral_screen')

# ======================================================================
# Evaluate top 10 + bottom 5
# ======================================================================
print('\\n=== Evaluate Top 10 + Bottom 5 ===', flush=True)
sbuid_data.sort(key=lambda x: x['sbuid_6k'], reverse=True)
to_eval = sbuid_data[:10] + sbuid_data[-5:]

for entry in to_eval:
    block = [tuple(entry['block'])]
    apply_blocks(block)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    restore()
    entry['math'] = math_r['score']
    entry['eq'] = eq_r['score']
    entry['combined'] = combined
    delta = combined - baseline
    print(f'  ({entry[\"block\"][0]},{entry[\"block\"][1]}): sbuid={entry[\"sbuid_6k\"]:7.0f} combined={combined:.2f} delta={delta:+.2f} ({time.time()-t0:.0f}s)', flush=True)

# Correlation
evaluated = [e for e in sbuid_data if 'combined' in e]
if len(evaluated) >= 5:
    for lam in ['sbuid_6k', 'sbuid_10k']:
        vals = [e[lam] for e in evaluated]
        combineds = [e['combined'] for e in evaluated]
        sr, sp = stats.spearmanr(vals, combineds)
        sig = '**' if sp < 0.05 else ''
        print(f'  {lam}: r={sr:.3f}, p={sp:.4f} {sig}', flush=True)

evaluated.sort(key=lambda x: x['combined'], reverse=True)
print(f'\\nTop 3:', flush=True)
for r in evaluated[:3]:
    print(f'  ({r[\"block\"][0]},{r[\"block\"][1]}): combined={r[\"combined\"]:.2f} delta={r[\"combined\"]-baseline:+.2f}', flush=True)

save({'baseline': baseline, 'evaluated': evaluated, 'all_sbuid': sbuid_data}, 'evaluated_blocks')

# ======================================================================
# Best block: attn-only vs full + gate margin
# ======================================================================
best = evaluated[0]
best_block = [tuple(best['block'])]
best_idx = best['block'][0]
print(f'\\n=== Sublayer Analysis: L{best_idx} ===', flush=True)

# Attn-only
apply_blocks(best_block)
module = original_layers[best_idx]
ctr = [0]
def make_zero(c):
    def hook(mod, inp, out):
        c[0] += 1
        if c[0] % 2 == 0:
            return (0.0 * out[0],) + out[1:] if isinstance(out, tuple) else 0.0 * out
        return out
    return hook
h = module.mlp.register_forward_hook(make_zero(ctr))
math_r = run_math_probe(gen, verbose=False)
eq_r = run_eq_bench_probe(gen_long, verbose=False)
attn_only = math_r['score'] * 50 + eq_r['score'] * 0.5
h.remove(); restore()
print(f'  attn_only: {attn_only:.2f}', flush=True)
print(f'  full_dup:  {best[\"combined\"]:.2f}', flush=True)
print(f'  FFN impact: {attn_only - best[\"combined\"]:+.2f}', flush=True)

# Gate margin
apply_blocks(best_block)
gate_proj = module.mlp.gate_proj
flips = []
for prompt in cal_prompts:
    gv, ctr2 = [], [0]
    def make_cap(c, g, gp):
        def hook(mod, inp, out):
            c[0] += 1
            with torch.no_grad():
                i = inp[0] if isinstance(inp, tuple) else inp
                g.append(gp(i[:, -1, :]).cpu().float().squeeze(0))
            return out
        return hook
    h = module.mlp.register_forward_hook(make_cap(ctr2, gv, gate_proj))
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
    with torch.no_grad(): model(inputs['input_ids'], use_cache=False)
    h.remove()
    if len(gv) >= 2:
        flips.append(((gv[0] > 0) != (gv[1] > 0)).float().mean().item())
restore()
avg_flip = float(np.mean(flips))
print(f'  gate_flip: {avg_flip:.4f}', flush=True)

save({
    'best_block': best['block'], 'full_dup': best['combined'],
    'attn_only': attn_only, 'ffn_impact': attn_only - best['combined'],
    'gate_flip': avg_flip,
}, 'sublayer_analysis')

# ======================================================================
# Summary
# ======================================================================
print(f'\\n{\"=\" * 50}', flush=True)
print(f'SUMMARY — LLaMA 3 8B ({N} layers)', flush=True)
print(f'Baseline: {baseline:.2f}', flush=True)
print(f'Best single: ({best[\"block\"][0]},{best[\"block\"][1]}) = {best[\"combined\"]:.2f} (+{best[\"combined\"]-baseline:.2f})', flush=True)
print(f'Attn-only: {attn_only:.2f} (FFN impact: {attn_only - best[\"combined\"]:+.2f})', flush=True)
print(f'Gate flip: {avg_flip:.4f}', flush=True)
if len(evaluated) >= 5:
    sr, sp = stats.spearmanr([e['sbuid_6k'] for e in evaluated], [e['combined'] for e in evaluated])
    print(f'SBUID r={sr:.3f}, p={sp:.4f}', flush=True)
print('COMPLETE', flush=True)
"

echo "=== Finished: $(date) ==="
