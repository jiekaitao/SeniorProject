#!/usr/bin/env python3
"""
Universal SBUID + Tier 2 sweep for any HuggingFace model.
Usage: python universal_sweep.py --model <hf_id_or_path> --name <short_name> [--cache_dir <dir>]
"""
import sys, os, json, time, argparse, torch, torch.nn as nn, numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe, _load_questions
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='HF model ID or local path')
    p.add_argument('--name', required=True, help='Short name for results dir')
    p.add_argument('--cache_dir', default='/blue/cis4914/jietao/hf_cache')
    p.add_argument('--max_blocks', type=int, default=15, help='Top blocks to evaluate')
    return p.parse_args()

args = parse_args()
SAVE_DIR = f'results/data/{args.name}'
os.makedirs(SAVE_DIR, exist_ok=True)

# Load model
print(f'Loading {args.model}...', flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model, cache_dir=args.cache_dir, device_map='auto',
    dtype=torch.bfloat16, trust_remote_code=True,
)
print(f'Loaded in {time.time()-t0:.0f}s', flush=True)

# Find layers - try multiple attribute paths
inner = None
layers_attr = None
for model_attr in ['model', 'transformer', 'backbone']:
    if hasattr(model, model_attr):
        candidate = getattr(model, model_attr)
        # Handle Gemma3 nesting
        if hasattr(candidate, 'language_model'):
            candidate = candidate.language_model
        for layer_attr in ['layers', 'h', 'blocks']:
            if hasattr(candidate, layer_attr):
                inner = candidate
                layers_attr = layer_attr
                break
        if inner: break

if inner is None:
    print(f'ERROR: Cannot find layers. Model type: {type(model).__name__}', flush=True)
    print(f'Attributes: {[a for a in dir(model) if not a.startswith("_")]}', flush=True)
    sys.exit(1)

original_layers = list(getattr(inner, layers_attr))
N = len(original_layers)
device = next(model.parameters()).device
print(f'{N} layers via {type(inner).__name__}.{layers_attr} on {device}', flush=True)

eq_all = _load_questions()
cal_prompts = [
    'What is 127 * 348?', 'What is 99999 * 99999?',
    'Calculate 15! / 13!', 'What is 2^16?',
    'What is the capital of France?', 'Explain entropy simply.',
]

def set_num_layers(n):
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg and hasattr(cfg, 'num_hidden_layers'):
            cfg.num_hidden_layers = n

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
    # Handle layer_types for Gemma models
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg and hasattr(cfg, 'layer_types') and cfg.layer_types:
            cfg.layer_types = [cfg.layer_types[idx] for idx in order]
        if cfg and hasattr(cfg, 'use_cache'):
            cfg.use_cache = False

def restore():
    setattr(inner, layers_attr, nn.ModuleList(original_layers))
    set_num_layers(N)

def gen_no_cache(prompt, max_tokens=64):
    inp = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)['input_ids'].to(device)
    for _ in range(max_tokens):
        with torch.no_grad():
            out = model(inp, use_cache=False)
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        if nxt.item() == tokenizer.eos_token_id: break
        inp = torch.cat([inp, nxt], dim=-1)
    plen = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)['input_ids'].shape[1]
    return tokenizer.decode(inp[0, plen:], skip_special_tokens=True)

def gen(p): return gen_no_cache(p, 64)
def gen_long(p): return gen_no_cache(p, 128)

def save(data, name):
    with open(f'{SAVE_DIR}/{name}.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  [SAVED] {name}.json', flush=True)

# ======================================================================
# BASELINE
# ======================================================================
print(f'\n=== Baseline ===', flush=True)
t0 = time.time()
math_r = run_math_probe(gen, verbose=False)
eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
baseline = math_r['score'] * 50 + eq_r['score'] * 0.5
print(f'Baseline: math={math_r["score"]:.4f} eq={eq_r["score"]:.1f} combined={baseline:.2f} ({time.time()-t0:.0f}s)', flush=True)
save({'math': math_r['score'], 'eq': eq_r['score'], 'combined': baseline, 'n_layers': N}, 'baseline')

# ======================================================================
# TIER 1: SBUID SCREEN
# ======================================================================
print(f'\n=== SBUID Spectral Screen ({N-1} blocks) ===', flush=True)
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
            setattr(inner, layers_attr, nn.ModuleList([original_layers[idx] for idx in order]))
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
    sbuid_data.append({
        'block': [start, start+1], 'rho': mr, 'blood': mb,
        'sbuid_6k': mb - 6000*mr, 'sbuid_10k': mb - 10000*mr, 'sbuid_20k': mb - 20000*mr,
    })
    if start % max(1, N//8) == 0:
        print(f'  L{start}: rho={mr:.4f} blood={mb:.1f} sbuid_6k={mb-6000*mr:.0f}', flush=True)

print(f'Screened {len(sbuid_data)} blocks', flush=True)
save(sbuid_data, 'spectral_screen')

# ======================================================================
# EVALUATE TOP + BOTTOM
# ======================================================================
print(f'\n=== Evaluate Top {args.max_blocks} + Bottom 5 ===', flush=True)
sbuid_data.sort(key=lambda x: x['sbuid_6k'], reverse=True)
to_eval = sbuid_data[:args.max_blocks] + sbuid_data[-5:]

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
    print(f'  ({entry["block"][0]},{entry["block"][1]}): sbuid={entry["sbuid_6k"]:7.0f} combined={combined:.2f} delta={delta:+.2f} ({time.time()-t0:.0f}s)', flush=True)

# SBUID Correlation
evaluated = [e for e in sbuid_data if 'combined' in e]
correlations = {}
if len(evaluated) >= 5:
    combineds = [e['combined'] for e in evaluated]
    for lam in ['sbuid_6k', 'sbuid_10k', 'sbuid_20k', 'rho', 'blood']:
        vals = [e[lam] for e in evaluated]
        sr, sp = stats.spearmanr(vals, combineds)
        sig = '***' if sp < 0.01 else ('**' if sp < 0.05 else ('*' if sp < 0.1 else ''))
        correlations[lam] = {'r': float(sr), 'p': float(sp)}
        print(f'  {lam}: r={sr:.3f}, p={sp:.4f} {sig}', flush=True)

evaluated.sort(key=lambda x: x['combined'], reverse=True)
print(f'\nTop 3:', flush=True)
for r in evaluated[:3]:
    print(f'  ({r["block"][0]},{r["block"][1]}): combined={r["combined"]:.2f} delta={r["combined"]-baseline:+.2f}', flush=True)

save({'baseline': baseline, 'evaluated': evaluated, 'correlations': correlations}, 'evaluated_blocks')

# ======================================================================
# TIER 2: SUBLAYER ANALYSIS ON BEST BLOCK
# ======================================================================
best = evaluated[0]
best_block = [tuple(best['block'])]
best_idx = best['block'][0]
print(f'\n=== Tier 2: Sublayer Analysis on L{best_idx} ===', flush=True)

# Find MLP submodule
module = original_layers[best_idx]
mlp_attr = None
for attr in ['mlp', 'feed_forward', 'ffn']:
    if hasattr(module, attr):
        mlp_attr = attr
        break

if mlp_attr is None:
    print(f'  WARNING: No MLP found. Submodules: {[n for n, _ in module.named_children()]}', flush=True)
    save({'best_block': best['block'], 'error': 'no MLP found'}, 'sublayer_analysis')
else:
    mlp_module = getattr(module, mlp_attr)

    # Attn-only
    apply_blocks(best_block)
    ctr = [0]
    def make_zero(c):
        def hook(mod, inp, out):
            c[0] += 1
            if c[0] % 2 == 0:
                return (0.0 * out[0],) + out[1:] if isinstance(out, tuple) else 0.0 * out
            return out
        return hook
    h = mlp_module.register_forward_hook(make_zero(ctr))
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    attn_only = math_r['score'] * 50 + eq_r['score'] * 0.5
    h.remove(); restore()
    print(f'  attn_only: {attn_only:.2f}', flush=True)

    # Whisper FFN
    apply_blocks(best_block)
    ctr2 = [0]
    def make_whisper(c):
        def hook(mod, inp, out):
            c[0] += 1
            if c[0] % 2 == 0:
                return (0.2 * out[0],) + out[1:] if isinstance(out, tuple) else 0.2 * out
            return out
        return hook
    h = mlp_module.register_forward_hook(make_whisper(ctr2))
    math_r2 = run_math_probe(gen, verbose=False)
    eq_r2 = run_eq_bench_probe(gen_long, verbose=False)
    whisper = math_r2['score'] * 50 + eq_r2['score'] * 0.5
    h.remove(); restore()
    print(f'  whisper02: {whisper:.2f}', flush=True)
    print(f'  full_dup:  {best["combined"]:.2f}', flush=True)
    print(f'  FFN impact: {attn_only - best["combined"]:+.2f}', flush=True)

    # Gate margin
    gate_proj = None
    for attr in ['gate_proj', 'w1', 'gate']:
        if hasattr(mlp_module, attr):
            gate_proj = getattr(mlp_module, attr)
            break

    avg_flip = -1
    if gate_proj is not None:
        apply_blocks(best_block)
        flips = []
        for prompt in cal_prompts:
            gv, ctr3 = [], [0]
            def make_cap(c, g, gp):
                def hook(mod, inp, out):
                    c[0] += 1
                    with torch.no_grad():
                        i = inp[0] if isinstance(inp, tuple) else inp
                        g.append(gp(i[:, -1, :]).cpu().float().squeeze(0))
                    return out
                return hook
            h = mlp_module.register_forward_hook(make_cap(ctr3, gv, gate_proj))
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
            with torch.no_grad(): model(inputs['input_ids'], use_cache=False)
            h.remove()
            if len(gv) >= 2:
                flips.append(((gv[0] > 0) != (gv[1] > 0)).float().mean().item())
        restore()
        avg_flip = float(np.mean(flips)) if flips else -1
        print(f'  gate_flip: {avg_flip:.4f}', flush=True)
    else:
        print(f'  gate_proj not found, skipping gate margin', flush=True)
        restore()

    save({
        'best_block': best['block'],
        'full_dup': best['combined'],
        'attn_only': attn_only,
        'whisper_ffn02': whisper,
        'ffn_impact': attn_only - best['combined'],
        'gate_flip': avg_flip,
    }, 'sublayer_analysis')

# ======================================================================
# SUMMARY
# ======================================================================
print(f'\n{"=" * 50}', flush=True)
print(f'SUMMARY — {args.name} ({N} layers)', flush=True)
print(f'Baseline: {baseline:.2f}', flush=True)
print(f'Best: ({best["block"][0]},{best["block"][1]}) = {best["combined"]:.2f} (+{best["combined"]-baseline:.2f})', flush=True)
if mlp_attr:
    print(f'Attn-only: {attn_only:.2f} (FFN impact: {attn_only - best["combined"]:+.2f})', flush=True)
    print(f'Whisper: {whisper:.2f}', flush=True)
    if avg_flip >= 0: print(f'Gate flip: {avg_flip:.4f}', flush=True)
if correlations:
    best_lam = max(correlations, key=lambda k: abs(correlations[k]['r']))
    print(f'SBUID ({best_lam}): r={correlations[best_lam]["r"]:.3f}, p={correlations[best_lam]["p"]:.4f}', flush=True)
print('COMPLETE', flush=True)
