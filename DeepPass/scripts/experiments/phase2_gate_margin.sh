#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_phase2_gate_margin_%j.log
#SBATCH --job-name=dp_gate

# Phase 2: Gate Margin Gating
# Tests the self-calibrating approach: only allow FFN neurons where the gate
# margin is large enough that the second pass won't flip the activation.
# If gate_first * gate_second > 0, the neuron fires the same way → safe.
# Tests multiple thresholds and evaluates on dual probe.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Phase 2: Gate Margin Gating ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe, _load_questions

MODEL_PATH = 'models/full/gemma-3-27b-it'
BLOCKS = [(0, 2), (12, 13), (47, 48)]
SAVE_DIR = 'results/data/gemma3_27b/neuron_analysis'
os.makedirs(SAVE_DIR, exist_ok=True)

print('Loading model...', flush=True)
model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

eq_all = _load_questions()

def set_num_layers(n):
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = n
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

def restore():
    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

# Duplicated layers
sorted_blocks = sorted(BLOCKS)
dup_layers = []
for (i, j) in sorted_blocks:
    for l in range(i, j):
        dup_layers.append(l)

# ======================================================================
# First: Measure gate margin statistics
# ======================================================================
print('\\n=== Measuring Gate Margins ===', flush=True)

# Apply duplication
order = build_order(BLOCKS, N)
inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
set_num_layers(len(order))

test_prompts = [
    'What is 127 * 348?',
    'What is the capital of France?',
    'Explain quantum entanglement simply.',
    'What is 2^16?',
    'Who wrote Romeo and Juliet?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'What is the chemical symbol for gold?',
    'What is the square root of 152399025?',
]

margin_stats = {l: {'margins': [], 'flip_rates': []} for l in dup_layers}

for layer_idx in dup_layers:
    module = original_layers[layer_idx]
    gate_proj = module.mlp.gate_proj

    for prompt in test_prompts:
        gate_vals = []
        counter = [0]

        def make_gate_capture(ctr, gvals, gproj):
            def hook(module, input, output):
                ctr[0] += 1
                with torch.no_grad():
                    inp = input[0] if isinstance(input, tuple) else input
                    # Capture raw gate values (pre-SiLU)
                    g = gproj(inp[:, -1, :]).cpu().float()
                    gvals.append(g.squeeze(0))
                return output
            return hook

        h = module.mlp.register_forward_hook(make_gate_capture(counter, gate_vals, gate_proj))
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).to(device)
        with torch.no_grad():
            model(inputs['input_ids'], use_cache=False)
        h.remove()

        if len(gate_vals) >= 2:
            g1 = gate_vals[0]  # first pass gate values
            g2 = gate_vals[1]  # second pass gate values

            # Gate margin = |g| (distance from decision boundary at 0 for SiLU)
            margins_1 = torch.abs(g1)
            margins_2 = torch.abs(g2)

            # Flip detection: sign change means different neuron activation
            flips = ((g1 > 0) != (g2 > 0)).float()
            flip_rate = flips.mean().item()

            # Agreement score: g1 * g2 > 0 means same activation
            agreement = (g1 * g2 > 0).float()

            margin_stats[layer_idx]['margins'].append(margins_1.mean().item())
            margin_stats[layer_idx]['flip_rates'].append(flip_rate)

    avg_margin = np.mean(margin_stats[layer_idx]['margins'])
    avg_flip = np.mean(margin_stats[layer_idx]['flip_rates'])
    print(f'  Layer {layer_idx}: avg_margin={avg_margin:.4f} avg_flip_rate={avg_flip:.4f}', flush=True)

restore()

# ======================================================================
# Gate Margin Gating: test various thresholds
# ======================================================================
print('\\n=== Gate Margin Gating Evaluation ===', flush=True)

# For each threshold, apply duplication with gate-margin-based FFN masking
thresholds = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
# Also test: attn_only (baseline), full dup, and uniform whisper for comparison
configs = [
    ('attn_only (ffn=0)', None, 0.0),
    ('whisper (ffn=0.2)', None, 0.2),
    ('full (ffn=1.0)', None, 1.0),
] + [
    (f'gate_margin_t={t}', t, None) for t in thresholds
]

results_list = []

for name, threshold, fixed_beta in configs:
    print(f'\\nConfig: {name}', flush=True)

    order = build_order(BLOCKS, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

    hooks = []
    for layer_idx in dup_layers:
        module = original_layers[layer_idx]
        gate_proj = module.mlp.gate_proj

        if threshold is not None:
            # Gate margin gating: capture first-pass gates, mask second-pass
            first_pass_gates = [None]
            counter = [0]

            def make_margin_hook(ctr, fpg, gproj, thresh):
                def hook(module, input, output):
                    ctr[0] += 1
                    inp = input[0] if isinstance(input, tuple) else input

                    if ctr[0] % 2 == 1:  # first pass: capture gates
                        with torch.no_grad():
                            fpg[0] = gproj(inp).detach()  # [batch, seq, d_ffn]
                    else:  # second pass: mask based on margin
                        with torch.no_grad():
                            second_gates = gproj(inp)  # [batch, seq, d_ffn]
                            # Agreement: both positive or both negative
                            agreement = (fpg[0] * second_gates > 0).float()
                            # Margin: minimum of both gate magnitudes
                            min_margin = torch.minimum(fpg[0].abs(), second_gates.abs())
                            # Keep neuron if agreement AND margin > threshold
                            mask = agreement * (min_margin > thresh).float()

                        # Scale output by mask (per-neuron)
                        if isinstance(output, tuple):
                            # output[0] is [batch, seq, d_model] — post down-projection
                            # We need to mask BEFORE down-projection, but we're hooking post-MLP
                            # So instead, mask the entire MLP output proportionally
                            keep_ratio = mask.mean(dim=-1, keepdim=True)  # [batch, seq, 1]
                            return (keep_ratio * output[0],) + output[1:]
                        else:
                            keep_ratio = mask.mean(dim=-1, keepdim=True)
                            return keep_ratio * output
                    return output
                return hook

            h = module.mlp.register_forward_hook(
                make_margin_hook(counter, first_pass_gates, gate_proj, threshold)
            )
            hooks.append(h)
        elif fixed_beta is not None and abs(fixed_beta - 1.0) > 1e-6:
            # Fixed beta scaling
            counter = [0]
            beta = fixed_beta

            def make_beta_hook(ctr, b):
                def hook(module, input, output):
                    ctr[0] += 1
                    if ctr[0] % 2 == 0:
                        if isinstance(output, tuple):
                            return (b * output[0],) + output[1:]
                        return b * output
                    return output
                return hook

            h = module.mlp.register_forward_hook(make_beta_hook(counter, beta))
            hooks.append(h)

    # Evaluate
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0

    print(f'  math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)

    results_list.append({
        'name': name,
        'threshold': threshold,
        'fixed_beta': fixed_beta,
        'math': math_r['score'],
        'eq': eq_r['score'],
        'combined': combined,
    })

    for h in hooks:
        h.remove()
    restore()

# ======================================================================
# SAVE
# ======================================================================
results_list.sort(key=lambda x: x['combined'], reverse=True)

print(f'\\n{\"=\" * 60}', flush=True)
print('RESULTS RANKED:', flush=True)
for r in results_list:
    print(f'  {r[\"name\"]:30s}: combined={r[\"combined\"]:.2f} math={r[\"math\"]:.4f} eq={r[\"eq\"]:.1f}', flush=True)
print(f'\\nBest: {results_list[0][\"name\"]} = {results_list[0][\"combined\"]:.2f}', flush=True)

with open(f'{SAVE_DIR}/phase2_gate_margin.json', 'w') as f:
    json.dump({
        'margin_stats': {str(k): v for k, v in margin_stats.items()},
        'results': results_list,
        'best': results_list[0],
    }, f, indent=2)
print(f'Saved to {SAVE_DIR}/phase2_gate_margin.json', flush=True)
"

echo "=== Finished: $(date) ==="
