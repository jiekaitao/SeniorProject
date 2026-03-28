#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_llama3_70b_spectral_%j.log
#SBATCH --job-name=ll3_spec

# LLaMA 3 70B — Proper spectral screening pipeline
# Phase 1: SBUID spectral screen ALL blocks (~20 min)
# Phase 2: Dual-probe evaluate top 15 candidates (~1.5h)
# Phase 3: Greedy pair stacking from best single
# Phase 4: Attn-only vs full + gate margin on best block
# Phase 5: Validate SBUID correlation (does it transfer to LLaMA?)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== LLaMA 3 70B — Spectral Screening Pipeline ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from scipy import stats

sys.path.insert(0, 'scripts')
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe, _load_questions

MODEL_PATH = '/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-70B-Instruct-hf'
SAVE_DIR = 'results/data/llama3_70b'
os.makedirs(SAVE_DIR, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

print('Loading LLaMA 3 70B...', flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map='auto', dtype=torch.bfloat16,
)
print(f'Loaded in {time.time()-t0:.0f}s', flush=True)

inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'{N} layers on {device}', flush=True)

eq_all = _load_questions()

def set_num_layers(n):
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
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

def restore():
    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

def gen(p):
    inp = tokenizer(p, return_tensors='pt')['input_ids'].to(device)
    for _ in range(64):
        with torch.no_grad():
            out = model(inp, use_cache=False)
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        if nxt.item() == tokenizer.eos_token_id:
            break
        inp = torch.cat([inp, nxt], dim=-1)
    plen = tokenizer(p, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(inp[0, plen:], skip_special_tokens=True)

def gen_long(p):
    inp = tokenizer(p, return_tensors='pt')['input_ids'].to(device)
    for _ in range(128):
        with torch.no_grad():
            out = model(inp, use_cache=False)
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        if nxt.item() == tokenizer.eos_token_id:
            break
        inp = torch.cat([inp, nxt], dim=-1)
    plen = tokenizer(p, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(inp[0, plen:], skip_special_tokens=True)

def save_checkpoint(data, name):
    with open(f'{SAVE_DIR}/{name}.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  [SAVED] {name}.json', flush=True)

# Calibration prompts for spectral analysis
cal_prompts = [
    'What is 127 * 348?',
    'What is 99999 * 99999?',
    'Calculate 15! / 13!',
    'What is 2^16?',
    'What is the sum of all integers from 1 to 100?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'What emotion would someone feel after losing a close friend?',
    'How would a parent feel seeing their child graduate?',
]

# ======================================================================
# Phase 0: Baseline
# ======================================================================
print('\\n=== Phase 0: Baseline ===', flush=True)
t0 = time.time()
math_base = run_math_probe(gen, verbose=False)
eq_base = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={baseline:.2f} ({time.time()-t0:.0f}s)', flush=True)
save_checkpoint({'math': math_base['score'], 'eq': eq_base['score'], 'combined': baseline}, 'baseline')

# ======================================================================
# Phase 1: SBUID Spectral Screen — ALL single blocks
# ======================================================================
print('\\n=== Phase 1: SBUID Spectral Screen ===', flush=True)

sbuid_data = []
for size in [1, 2, 3]:
    for start in range(0, N - size, 1 if size == 1 else 2):
        block = (start, start + size)
        rhos = []
        blood_impacts = []

        for prompt in cal_prompts[:4]:
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
            with torch.no_grad():
                # Baseline forward
                out_base = model(inputs['input_ids'], use_cache=False)
                logits_base = out_base.logits[:, -1, :].float()

                # Capture hidden state at block exit for BLOOD
                hidden_before = [None]
                hidden_after = [None]
                counter = [0]

                def make_blood_hook(ctr, hb, ha):
                    def hook(module, input, output):
                        ctr[0] += 1
                        h = output[0] if isinstance(output, tuple) else output
                        if ctr[0] % 2 == 1:
                            hb[0] = h.detach().float()
                        else:
                            ha[0] = h.detach().float()
                        return output
                    return hook

                hook = original_layers[start].register_forward_hook(
                    make_blood_hook(counter, hidden_before, hidden_after)
                )

                # Duplicated forward
                order = build_order([block], N)
                inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
                set_num_layers(len(order))

                out_dup = model(inputs['input_ids'], use_cache=False)
                logits_dup = out_dup.logits[:, -1, :].float()

                hook.remove()
                restore()

                # Displacement rho
                num = torch.norm(logits_dup - logits_base).item()
                den = torch.norm(logits_base).item()
                if den > 1e-8:
                    rhos.append(num / den)

                # BLOOD impact
                if hidden_before[0] is not None and hidden_after[0] is not None:
                    diff = torch.norm(hidden_after[0] - hidden_before[0]).item()
                    blood_impacts.append(diff)

        mean_rho = float(np.mean(rhos)) if rhos else 1.0
        mean_blood = float(np.mean(blood_impacts)) if blood_impacts else 0.0

        # Try multiple lambdas — will calibrate later
        sbuid_6k = mean_blood - 6000 * mean_rho
        sbuid_10k = mean_blood - 10000 * mean_rho
        sbuid_20k = mean_blood - 20000 * mean_rho

        sbuid_data.append({
            'block': list(block), 'start': start, 'size': size,
            'rho': mean_rho, 'blood': mean_blood,
            'sbuid_6k': sbuid_6k, 'sbuid_10k': sbuid_10k, 'sbuid_20k': sbuid_20k,
        })

        if start % 10 == 0 and size == 1:
            print(f'  ({start},{start+size}): rho={mean_rho:.4f} blood={mean_blood:.1f} sbuid={sbuid_6k:.0f}', flush=True)

print(f'Screened {len(sbuid_data)} blocks', flush=True)
save_checkpoint(sbuid_data, 'spectral_screen')

# ======================================================================
# Phase 2: Dual-probe top 15 by SBUID
# ======================================================================
print('\\n=== Phase 2: Evaluate Top 15 by SBUID ===', flush=True)

# Sort by SBUID (lambda=6000, the 72B-validated value)
sbuid_data.sort(key=lambda x: x['sbuid_6k'], reverse=True)

# Also evaluate bottom 5 for correlation validation
to_eval = sbuid_data[:15] + sbuid_data[-5:]

eval_results = []
for entry in to_eval:
    block = [tuple(entry['block'])]
    apply_blocks(block)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    restore()

    entry['math'] = math_r['score']
    entry['eq'] = eq_r['score']
    entry['combined'] = combined
    delta = combined - baseline
    print(f'  ({entry[\"block\"][0]},{entry[\"block\"][1]}): sbuid={entry[\"sbuid_6k\"]:7.0f} combined={combined:.2f} delta={delta:+.2f} ({elapsed:.0f}s)', flush=True)
    eval_results.append(entry)

eval_results.sort(key=lambda x: x['combined'], reverse=True)
print(f'\\nTop 5 blocks:', flush=True)
for r in eval_results[:5]:
    print(f'  ({r[\"block\"][0]},{r[\"block\"][1]}): combined={r[\"combined\"]:.2f} delta={r[\"combined\"]-baseline:+.2f}', flush=True)

save_checkpoint({'baseline': baseline, 'evaluated': eval_results}, 'evaluated_blocks')

# ======================================================================
# Phase 2b: SBUID Correlation Validation
# ======================================================================
print('\\n=== Phase 2b: SBUID Correlation ===', flush=True)

evaluated = [e for e in sbuid_data if 'combined' in e]
if len(evaluated) >= 5:
    for lam_name in ['sbuid_6k', 'sbuid_10k', 'sbuid_20k']:
        sbuids = [e[lam_name] for e in evaluated]
        combineds = [e['combined'] for e in evaluated]
        sr, sp = stats.spearmanr(sbuids, combineds)
        sig = '***' if sp < 0.01 else ('**' if sp < 0.05 else ('*' if sp < 0.1 else ''))
        print(f'  {lam_name}: Spearman r={sr:.3f}, p={sp:.4f} {sig}', flush=True)

    # Also check raw components
    rhos = [e['rho'] for e in evaluated]
    bloods = [e['blood'] for e in evaluated]
    sr_r, sp_r = stats.spearmanr(rhos, combineds)
    sr_b, sp_b = stats.spearmanr(bloods, combineds)
    print(f'  rho alone: r={sr_r:.3f}, p={sp_r:.4f}', flush=True)
    print(f'  BLOOD alone: r={sr_b:.3f}, p={sp_b:.4f}', flush=True)

save_checkpoint({
    'n_evaluated': len(evaluated),
    'correlations': {
        'sbuid_6k': {'r': float(sr), 'p': float(sp)} if len(evaluated) >= 5 else None,
    },
    'evaluated': evaluated,
}, 'sbuid_validation')

# ======================================================================
# Phase 3: Greedy Pair Stacking
# ======================================================================
print('\\n=== Phase 3: Greedy Pair from Best Single ===', flush=True)

best_single = eval_results[0]
anchor = tuple(best_single['block'])
print(f'Anchor: ({anchor[0]},{anchor[1]}) combined={best_single[\"combined\"]:.2f}', flush=True)

# Screen complementary blocks: re-apply anchor, then spectral screen for 2nd block
apply_blocks([anchor])

pair_candidates = []
for entry in sbuid_data[:25]:  # top 25 by SBUID
    cand = tuple(entry['block'])
    if not (cand[1] <= anchor[0] or anchor[1] <= cand[0]):
        continue  # overlaps
    blocks = [anchor, cand]
    apply_blocks(blocks)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    restore()

    delta = combined - best_single['combined']
    print(f'  +({cand[0]},{cand[1]}): combined={combined:.2f} delta_vs_single={delta:+.2f} ({elapsed:.0f}s)', flush=True)
    pair_candidates.append({
        'blocks': [list(anchor), list(cand)],
        'math': math_r['score'], 'eq': eq_r['score'],
        'combined': combined, 'delta_vs_single': delta,
    })

pair_candidates.sort(key=lambda x: x['combined'], reverse=True)
print(f'\\nTop 5 pairs:', flush=True)
for r in pair_candidates[:5]:
    b = '+'.join(f'({b[0]},{b[1]})' for b in r['blocks'])
    print(f'  {b}: combined={r[\"combined\"]:.2f}', flush=True)

save_checkpoint({'anchor': best_single, 'pairs': pair_candidates}, 'pair_search')

# ======================================================================
# Phase 4: Sublayer Analysis on Best Block
# ======================================================================
print('\\n=== Phase 4: Attn-Only vs Full + Gate Margin ===', flush=True)

best_block = [tuple(best_single['block'])]
dup_layers = list(range(best_block[0][0], best_block[0][1]))

# Attn-only
apply_blocks(best_block)
hooks = []
for layer_idx in dup_layers:
    module = original_layers[layer_idx]
    counter = [0]
    def make_zero_ffn(ctr):
        def hook(mod, inp, out):
            ctr[0] += 1
            if ctr[0] % 2 == 0:
                if isinstance(out, tuple):
                    return (0.0 * out[0],) + out[1:]
                return 0.0 * out
            return out
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
print(f'  attn_only: combined={attn_combined:.2f} ({time.time()-t0:.0f}s)', flush=True)

# Whisper FFN
apply_blocks(best_block)
hooks = []
for layer_idx in dup_layers:
    module = original_layers[layer_idx]
    counter = [0]
    def make_whisper(ctr):
        def hook(mod, inp, out):
            ctr[0] += 1
            if ctr[0] % 2 == 0:
                if isinstance(out, tuple):
                    return (0.2 * out[0],) + out[1:]
                return 0.2 * out
            return out
        return hook
    h = module.mlp.register_forward_hook(make_whisper(counter))
    hooks.append(h)

t0 = time.time()
math_r2 = run_math_probe(gen, verbose=False)
eq_r2 = run_eq_bench_probe(gen_long, verbose=False)
whisper_combined = math_r2['score'] * 50 + eq_r2['score'] * 0.5
for h in hooks:
    h.remove()
restore()
print(f'  whisper02: combined={whisper_combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
print(f'  full_dup:  combined={best_single[\"combined\"]:.2f}', flush=True)
print(f'  FFN impact (attn_only - full): {attn_combined - best_single[\"combined\"]:+.2f}', flush=True)

# Gate margin
print('\\n  Gate margins:', flush=True)
gate_stats = {}
for layer_idx in dup_layers:
    apply_blocks(best_block)
    module = original_layers[layer_idx]
    gate_proj = module.mlp.gate_proj

    flip_rates = []
    for prompt in cal_prompts[:6]:
        gate_vals = []
        counter = [0]
        def make_capture(ctr, gvals, gp):
            def hook(mod, inp, out):
                ctr[0] += 1
                with torch.no_grad():
                    i = inp[0] if isinstance(inp, tuple) else inp
                    gvals.append(gp(i[:, -1, :]).cpu().float().squeeze(0))
                return out
            return hook
        h = module.mlp.register_forward_hook(make_capture(counter, gate_vals, gate_proj))
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).to(device)
        with torch.no_grad():
            model(inputs['input_ids'], use_cache=False)
        h.remove()
        if len(gate_vals) >= 2:
            flips = ((gate_vals[0] > 0) != (gate_vals[1] > 0)).float().mean().item()
            flip_rates.append(flips)

    restore()
    avg_flip = float(np.mean(flip_rates)) if flip_rates else -1
    gate_stats[str(layer_idx)] = avg_flip
    stab = 'stable' if avg_flip < 0.05 else ('moderate' if avg_flip < 0.2 else 'unstable')
    print(f'    Layer {layer_idx}: flip_rate={avg_flip:.4f} ({stab})', flush=True)

save_checkpoint({
    'best_block': list(best_block[0]),
    'full_dup': best_single['combined'],
    'attn_only': {'math': math_r['score'], 'eq': eq_r['score'], 'combined': attn_combined},
    'whisper_ffn02': {'math': math_r2['score'], 'eq': eq_r2['score'], 'combined': whisper_combined},
    'ffn_impact': attn_combined - best_single['combined'],
    'gate_margins': gate_stats,
}, 'sublayer_analysis')

# ======================================================================
# Summary
# ======================================================================
print(f'\\n{\"=\" * 60}', flush=True)
print('SUMMARY — LLaMA 3 70B', flush=True)
print(f'{\"=\" * 60}', flush=True)
print(f'Baseline: {baseline:.2f}', flush=True)
print(f'Best single: ({best_single[\"block\"][0]},{best_single[\"block\"][1]}) = {best_single[\"combined\"]:.2f} (+{best_single[\"combined\"]-baseline:.2f})', flush=True)
if pair_candidates:
    bp = pair_candidates[0]
    print(f'Best pair: {bp[\"blocks\"]} = {bp[\"combined\"]:.2f} (+{bp[\"combined\"]-baseline:.2f})', flush=True)
print(f'Attn-only: {attn_combined:.2f}', flush=True)
print(f'Whisper FFN: {whisper_combined:.2f}', flush=True)
print(f'FFN impact: {attn_combined - best_single[\"combined\"]:+.2f}', flush=True)
if len(evaluated) >= 5:
    print(f'SBUID correlation: r={sr:.3f}, p={sp:.4f}', flush=True)
print('COMPLETE', flush=True)
"

echo "=== Finished: $(date) ==="
