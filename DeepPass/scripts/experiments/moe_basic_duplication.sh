#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_moe_basic_%j.log
#SBATCH --job-name=deeppass_moeb

# Basic layer duplication on Qwen3-30B-A3B (MoE: 128 experts, 8 active/tok, 48 layers)
# Does RYS-style duplication work on a Mixture of Experts model?
# Also: does the router route to DIFFERENT experts on pass 2?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== MoE Basic Duplication: Qwen3-30B-A3B ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, gc, torch, torch.nn as nn, numpy as np
from datetime import datetime
from collections import defaultdict
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

print('=' * 70, flush=True)
print('MoE BASIC DUPLICATION: Qwen3-30B-A3B', flush=True)
print('128 experts, 8 active/tok, 48 layers, ~30B total / ~3B active', flush=True)
print('=' * 70, flush=True)

# =========================================================================
# Step 0: Load model and inspect MoE structure
# =========================================================================
print('\n=== Step 0: Load Model & Inspect MoE Architecture ===', flush=True)
model, tokenizer = load_original_model('models/full/Qwen3-30B-A3B')

inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers on {device}', flush=True)
print(f'VRAM used: ~{torch.cuda.memory_allocated()/1e9:.1f} GB', flush=True)

# Inspect MoE structure
print('\n--- MoE Structure Inspection ---', flush=True)
layer0 = inner.layers[0]
print(f'Layer 0 type: {type(layer0).__name__}', flush=True)
print(f'Layer 0 children:', flush=True)
for name, child in layer0.named_children():
    print(f'  .{name}: {type(child).__name__}', flush=True)

# Find the MoE block attribute name
moe_attr = None
gate_path = None
for name, child in layer0.named_children():
    child_type = type(child).__name__.lower()
    if 'sparse' in child_type or 'moe' in child_type:
        moe_attr = name
        print(f'  => MoE block at .{name}', flush=True)
        for sub_name, sub_child in child.named_children():
            print(f'     .{name}.{sub_name}: {type(sub_child).__name__}', flush=True)
            sub_type = type(sub_child).__name__.lower()
            if 'router' in sub_type or 'gate' in sub_type:
                gate_path = f'{name}.{sub_name}'
                # Check for gate linear layer inside router
                for sub2_name, sub2_child in sub_child.named_children():
                    print(f'       .{name}.{sub_name}.{sub2_name}: {type(sub2_child).__name__}', flush=True)
        break

# Fallback: check mlp for gate
if moe_attr is None:
    if hasattr(layer0, 'mlp'):
        mlp = layer0.mlp
        mlp_type = type(mlp).__name__.lower()
        print(f'  .mlp type: {type(mlp).__name__}', flush=True)
        if 'sparse' in mlp_type or 'moe' in mlp_type:
            moe_attr = 'mlp'
            for sub_name, sub_child in mlp.named_children():
                print(f'     .mlp.{sub_name}: {type(sub_child).__name__}', flush=True)
                sub_type = type(sub_child).__name__.lower()
                if 'router' in sub_type or 'gate' in sub_type:
                    gate_path = f'mlp.{sub_name}'
                    for sub2_name, sub2_child in sub_child.named_children():
                        print(f'       .mlp.{sub_name}.{sub2_name}: {type(sub2_child).__name__}', flush=True)

# Count which layers are MoE vs dense
moe_layers = []
dense_layers = []
for idx in range(N):
    layer = inner.layers[idx]
    is_moe = False
    for name, child in layer.named_children():
        child_type = type(child).__name__.lower()
        if 'sparse' in child_type or 'moe' in child_type:
            is_moe = True
            break
    if not is_moe and hasattr(layer, 'mlp'):
        mlp_type = type(layer.mlp).__name__.lower()
        if 'sparse' in mlp_type or 'moe' in mlp_type:
            is_moe = True
    if is_moe:
        moe_layers.append(idx)
    else:
        dense_layers.append(idx)

print(f'\nMoE layers: {len(moe_layers)} / {N}', flush=True)
print(f'Dense layers: {len(dense_layers)} / {N}', flush=True)
if dense_layers:
    print(f'Dense layer indices: {dense_layers}', flush=True)

def set_num_layers(n):
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = n
    else:
        model.config.num_hidden_layers = n

# =========================================================================
# Step 1: Quick spectral screen (architecture-agnostic, full-model forward)
# =========================================================================
print('\n=== Step 1: Spectral Screen (step=4, sizes=1,3,5,7) ===', flush=True)

cal_prompts = [
    'What is 127 * 348?', 'What is 99999 * 99999?',
    'Calculate 15! / 13!', 'What is 2^16?',
    'What is the sum of all integers from 1 to 100?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'What emotion would someone feel after losing a close friend?',
    'How would a parent feel seeing their child graduate?',
]

def compute_rho(block):
    i, j = block
    rhos = []
    for prompt in cal_prompts:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            # Base forward
            out_base = model(inputs['input_ids'], use_cache=False)
            logits_base = out_base.logits[:, -1, :].float()

            # Duplicated forward
            order = list(range(j)) + list(range(i, j)) + list(range(j, N))
            inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
            set_num_layers(len(order))

            out_dup = model(inputs['input_ids'], use_cache=False)
            logits_dup = out_dup.logits[:, -1, :].float()

            # Restore
            inner.layers = nn.ModuleList(original_layers)
            set_num_layers(N)

            num = torch.norm(logits_dup - logits_base).item()
            den = torch.norm(logits_base).item()
            if den > 1e-8:
                rhos.append(num / den)
    return float(np.mean(rhos)) if rhos else 1.0

candidates = []
for start in range(0, N - 1, 4):
    for size in [1, 3, 5, 7]:
        end = start + size
        if end <= N:
            candidates.append((start, end))

print(f'Screening {len(candidates)} blocks...', flush=True)
block_rhos = {}
for idx, block in enumerate(candidates):
    block_rhos[block] = compute_rho(block)
    if (idx + 1) % 5 == 0:
        print(f'  [{idx+1}/{len(candidates)}] ({block[0]:2d},{block[1]:2d}) size={block[1]-block[0]} rho={block_rhos[block]:.4f}', flush=True)

sorted_blocks = sorted(block_rhos.items(), key=lambda x: x[1])
print('\nTop 15 blocks by rho (lower = more contractive = better candidate):', flush=True)
for b, r in sorted_blocks[:15]:
    is_moe_block = any(l in moe_layers for l in range(b[0], b[1]))
    tag = 'MoE' if is_moe_block else 'dense'
    print(f'  ({b[0]:2d},{b[1]:2d}) size={b[1]-b[0]}: rho={r:.4f} [{tag}]', flush=True)

# =========================================================================
# Step 2: Baseline + top-8 singles with dual probe
# =========================================================================
print('\n=== Step 2: Baseline + Top-8 Singles (Dual Probe) ===', flush=True)

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

math_base = run_math_probe(gen, verbose=False)
eq_base = run_eq_bench_probe(gen_long, verbose=False)
combined_base = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={combined_base:.2f}', flush=True)

results = {
    'model': 'Qwen3-30B-A3B',
    'architecture': 'MoE',
    'num_layers': N,
    'num_experts': 128,
    'experts_per_tok': 8,
    'moe_layers': moe_layers,
    'dense_layers': dense_layers,
    'baseline': {'math': math_base['score'], 'eq': eq_base['score'], 'combined': combined_base},
}

single_scores = {}
top_8 = [b for b, r in sorted_blocks[:8]]

for block in top_8:
    i, j = block
    order = build_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    delta = combined - combined_base

    is_moe_block = any(l in moe_layers for l in range(i, j))
    tag = 'MoE' if is_moe_block else 'dense'
    print(f'  ({i:2d},{j:2d}) size={j-i} [{tag}]: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} delta={delta:+.2f}', flush=True)
    single_scores[block] = combined
    results[f'({i},{j})'] = {
        'block': list(block), 'rho': block_rhos[block],
        'math': math_r['score'], 'eq': eq_r['score'],
        'combined': combined, 'delta': delta,
        'is_moe': is_moe_block,
    }

    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

# =========================================================================
# Step 3: Greedy stacking — find best pair
# =========================================================================
print('\n=== Step 3: Greedy Stacking ===', flush=True)
best_single = max(single_scores, key=single_scores.get)
best_score = single_scores[best_single]
print(f'Best single: ({best_single[0]},{best_single[1]}) combined={best_score:.2f}', flush=True)

# Apply best single, screen for second block
order_a = build_order([best_single], N)
inner.layers = nn.ModuleList([original_layers[idx] for idx in order_a])
set_num_layers(len(order_a))

second_candidates = [b for b in [x[0] for x in sorted_blocks[:15]]
                     if b[1] <= best_single[0] or b[0] >= best_single[1]]

print(f'Screening {len(second_candidates)} second-block candidates on modified model...', flush=True)
second_rhos = {}
for block in second_candidates:
    second_rhos[block] = compute_rho(block)

inner.layers = nn.ModuleList(original_layers)
set_num_layers(N)

sorted_second = sorted(second_rhos.items(), key=lambda x: x[1])
print('Top 5 second blocks:', flush=True)
for b, r in sorted_second[:5]:
    print(f'  ({b[0]:2d},{b[1]:2d}): rho={r:.4f}', flush=True)

# Evaluate top-5 pairs
pair_results = []
for block_b, _ in sorted_second[:5]:
    pair = sorted([best_single, block_b])
    name = '+'.join(f'({b[0]},{b[1]})' for b in pair)

    order = build_order(pair, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    delta = combined - best_score

    print(f'  {name}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} vs best single: {delta:+.2f}', flush=True)
    pair_results.append({
        'blocks': [list(b) for b in pair], 'name': name,
        'math': math_r['score'], 'eq': eq_r['score'],
        'combined': combined, 'delta_vs_best_single': delta,
    })

    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

results['pairs'] = pair_results

# =========================================================================
# Step 4: Whisper-alpha triples (if pairs work)
# =========================================================================
print('\n=== Step 4: Whisper-Alpha Triples ===', flush=True)
best_pair_result = max(pair_results, key=lambda x: x['combined']) if pair_results else None

if best_pair_result and best_pair_result['combined'] > best_score:
    print('Pair beats single! Trying triples...', flush=True)
    bp_blocks = [tuple(b) for b in best_pair_result['blocks']]
    bp_score = best_pair_result['combined']

    # Screen for third block on double-modified model
    order_ab = build_order(bp_blocks, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order_ab])
    set_num_layers(len(order_ab))

    third_candidates = [b for b in [x[0] for x in sorted_blocks[:15]]
                        if all(b[1] <= blk[0] or b[0] >= blk[1] for blk in bp_blocks)]

    third_rhos = {}
    for block in third_candidates[:8]:
        third_rhos[block] = compute_rho(block)

    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

    sorted_third = sorted(third_rhos.items(), key=lambda x: x[1])

    triple_results = []
    for block_c, _ in sorted_third[:3]:
        triple = sorted(bp_blocks + [block_c])
        name = '+'.join(f'({b[0]},{b[1]})' for b in triple)

        order = build_order(triple, N)
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
        set_num_layers(len(order))

        math_r = run_math_probe(gen, verbose=False)
        eq_r = run_eq_bench_probe(gen_long, verbose=False)
        combined = math_r['score'] * 50 + eq_r['score'] * 0.5
        delta = combined - bp_score

        print(f'  {name}: combined={combined:.2f} vs best pair: {delta:+.2f}', flush=True)
        triple_results.append({
            'blocks': [list(b) for b in triple], 'name': name,
            'math': math_r['score'], 'eq': eq_r['score'],
            'combined': combined, 'delta_vs_best_pair': delta,
        })

        inner.layers = nn.ModuleList(original_layers)
        set_num_layers(N)

    results['triples'] = triple_results
else:
    print('Pair does not beat single. Skipping triples.', flush=True)
    results['triples'] = []

# =========================================================================
# Step 5: Router analysis — do experts change between pass 1 and pass 2?
# =========================================================================
print('\n=== Step 5: Router Analysis (Expert Switching Rate) ===', flush=True)
print('For the best duplicated block, compare router decisions pass 1 vs pass 2.', flush=True)

best_block_for_analysis = best_single
bi, bj = best_block_for_analysis
print(f'Analyzing block ({bi},{bj})...', flush=True)

# Find the gate/router module path for hooking
def find_gate_module(layer):
    \"\"\"Find the gate/router linear layer inside a transformer layer.\"\"\"
    # Try: layer.block_sparse_moe.router.gate (Qwen3MoE)
    # Try: layer.mlp.gate (some architectures)
    # Try: layer.block_sparse_moe.gate
    candidates = [
        ('block_sparse_moe', 'router', 'gate'),
        ('block_sparse_moe', 'gate'),
        ('mlp', 'router', 'gate'),
        ('mlp', 'gate'),
    ]
    for path in candidates:
        obj = layer
        found = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                found = False
                break
        if found and isinstance(obj, nn.Linear):
            return obj, '.'.join(path)
    # Brute-force search
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and 'gate' in name.lower():
            return mod, name
    return None, None

# Verify we can find gates
sample_gate, sample_path = find_gate_module(inner.layers[moe_layers[0]] if moe_layers else inner.layers[0])
if sample_gate is not None:
    print(f'Gate found at: layer.{sample_path} (shape: {list(sample_gate.weight.shape)})', flush=True)
else:
    print('WARNING: Could not find gate module. Trying alternative detection...', flush=True)
    # Print full module tree for first MoE layer
    target_idx = moe_layers[0] if moe_layers else 0
    for name, mod in inner.layers[target_idx].named_modules():
        if name:
            print(f'  {name}: {type(mod).__name__}', flush=True)

# Router analysis prompts
router_prompts = [
    'What is 127 * 348?',
    'Calculate the derivative of x^3 + 2x^2 - 5x + 1.',
    'What is the capital of France?',
    'How would a parent feel seeing their child graduate?',
    'What is 2^16?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
]

# Collect router decisions during pass 1 and pass 2 for the duplicated block
# Strategy: use hooks to capture gate outputs (pre-softmax logits) during forward
# The duplicated model runs layers: [0..bj-1, bi..bj-1, bj..N-1]
# Pass 1 = first occurrence of layers bi..bj-1 (positions bi..bj-1 in order)
# Pass 2 = second occurrence of layers bi..bj-1 (positions bj..bj+(bj-bi)-1 in order)

router_analysis = {}

if sample_gate is not None:
    for prompt_idx, prompt in enumerate(router_prompts):
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)

        # We need to track which call to each layer's gate is pass 1 vs pass 2
        # Since same layer module is reused, we track call count
        gate_outputs = defaultdict(list)  # layer_idx -> [pass1_topk, pass2_topk]
        call_counts = defaultdict(int)

        hooks = []
        for layer_idx in range(bi, bj):
            layer = original_layers[layer_idx]
            gate_mod, _ = find_gate_module(layer)
            if gate_mod is None:
                continue

            def make_hook(lidx):
                def hook_fn(module, input, output):
                    # output = gate logits, shape [batch, seq_len, num_experts]
                    with torch.no_grad():
                        # Get top-k expert indices
                        topk = torch.topk(output, k=8, dim=-1).indices  # [batch, seq, 8]
                        gate_outputs[lidx].append(topk.cpu())
                    call_counts[lidx] += 1
                return hook_fn
            hooks.append(gate_mod.register_forward_hook(make_hook(layer_idx)))

        # Forward through duplicated model
        order = list(range(bj)) + list(range(bi, bj)) + list(range(bj, N))
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
        set_num_layers(len(order))

        with torch.no_grad():
            model(inputs['input_ids'], use_cache=False)

        # Restore
        inner.layers = nn.ModuleList(original_layers)
        set_num_layers(N)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Analyze: for each layer in the block, compare pass 1 vs pass 2 expert selections
        switching_rates = []
        for layer_idx in range(bi, bj):
            if layer_idx in gate_outputs and len(gate_outputs[layer_idx]) >= 2:
                pass1_experts = gate_outputs[layer_idx][0]  # [batch, seq, 8]
                pass2_experts = gate_outputs[layer_idx][1]  # [batch, seq, 8]

                # Compare: for each token position, what fraction of selected experts changed?
                # Sort both for set comparison
                p1_sorted = torch.sort(pass1_experts, dim=-1).values
                p2_sorted = torch.sort(pass2_experts, dim=-1).values

                # Fraction of positions where expert sets differ
                match = (p1_sorted == p2_sorted).all(dim=-1).float()
                switching_rate = 1.0 - match.mean().item()
                switching_rates.append(switching_rate)

        avg_switching = float(np.mean(switching_rates)) if switching_rates else 0.0
        print(f'  Prompt {prompt_idx}: \"{prompt[:50]}...\" switching_rate={avg_switching:.4f} ({avg_switching*100:.1f}%)', flush=True)
        router_analysis[prompt] = {
            'switching_rate': avg_switching,
            'per_layer_switching': switching_rates,
        }

    # Summary
    overall_switching = float(np.mean([v['switching_rate'] for v in router_analysis.values()]))
    print(f'\n  OVERALL ROUTER SWITCHING RATE: {overall_switching:.4f} ({overall_switching*100:.1f}%)', flush=True)
    if overall_switching > 0.3:
        print('  => HIGH switching: router adapts significantly on pass 2!', flush=True)
        print('  => This means the MoE routing is INPUT-DEPENDENT even on duplicated pass.', flush=True)
    elif overall_switching > 0.1:
        print('  => MODERATE switching: some expert reassignment on pass 2.', flush=True)
    else:
        print('  => LOW switching: router picks ~same experts on both passes.', flush=True)
        print('  => Duplication acts more like a residual refinement within same experts.', flush=True)

    results['router_analysis'] = {
        'block': list(best_block_for_analysis),
        'overall_switching_rate': overall_switching,
        'per_prompt': {p: v for p, v in router_analysis.items()},
    }
else:
    print('Could not perform router analysis (gate module not found).', flush=True)
    results['router_analysis'] = {'error': 'gate module not found'}

# =========================================================================
# Summary
# =========================================================================
print('\n' + '=' * 70, flush=True)
print('SUMMARY: MoE Basic Duplication on Qwen3-30B-A3B', flush=True)
print('=' * 70, flush=True)
print(f'Model: Qwen3-30B-A3B ({N} layers, {len(moe_layers)} MoE / {len(dense_layers)} dense)', flush=True)
print(f'Baseline: math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={combined_base:.2f}', flush=True)

# Best single
print(f'\nBest single block: ({best_single[0]},{best_single[1]}) combined={best_score:.2f} delta={best_score-combined_base:+.2f}', flush=True)

# Best pair
if pair_results:
    bp = max(pair_results, key=lambda x: x['combined'])
    print(f'Best pair: {bp[\"name\"]} combined={bp[\"combined\"]:.2f} delta={bp[\"combined\"]-combined_base:+.2f}', flush=True)
    if bp['combined'] > best_score:
        print('*** PAIR BEATS SINGLE! ***', flush=True)
    else:
        print('Pair does not beat single (consistent with dense model findings).', flush=True)

# Key question
duplication_works = best_score > combined_base
print(f'\nDoes layer duplication work on MoE? {\"YES\" if duplication_works else \"NO\"} (delta={best_score-combined_base:+.2f})', flush=True)

# Spectral rankings
print(f'\nSpectral screen top-5:', flush=True)
for b, r in sorted_blocks[:5]:
    is_moe_block = any(l in moe_layers for l in range(b[0], b[1]))
    tag = 'MoE' if is_moe_block else 'dense'
    print(f'  ({b[0]:2d},{b[1]:2d}) rho={r:.4f} [{tag}]', flush=True)

# Save results
os.makedirs('results/data/moe', exist_ok=True)
results['spectral'] = [{'block': list(b), 'rho': r} for b, r in sorted_blocks]
results['date'] = datetime.now().isoformat()

with open('results/data/moe/basic_duplication.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\nSaved to results/data/moe/basic_duplication.json', flush=True)
"

echo "=== Done at $(date) ==="
