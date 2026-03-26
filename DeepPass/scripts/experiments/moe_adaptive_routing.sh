#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_moe_adaptive_%j.log
#SBATCH --job-name=deeppass_moea

# Router-guided adaptive layer duplication on Qwen3-30B-A3B
# Hypothesis: expert routing patterns reveal which layers to duplicate per input type.
# Layers where math inputs route to DIFFERENT experts than other inputs
# are the best candidates for math-specific duplication.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== MoE Adaptive Routing: Qwen3-30B-A3B ==="
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
print('MoE ADAPTIVE ROUTING: Qwen3-30B-A3B', flush=True)
print('Router-guided layer duplication selection', flush=True)
print('=' * 70, flush=True)

# =========================================================================
# Step 0: Load model and find MoE components
# =========================================================================
print('\n=== Step 0: Load Model ===', flush=True)
model, tokenizer = load_original_model('models/full/Qwen3-30B-A3B')

inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers on {device}', flush=True)
print(f'VRAM: ~{torch.cuda.memory_allocated()/1e9:.1f} GB', flush=True)

def set_num_layers(n):
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = n
    else:
        model.config.num_hidden_layers = n

# Find MoE gate modules
def find_gate_module(layer):
    \"\"\"Find the gate/router linear layer inside a transformer layer.\"\"\"
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
        if found and (isinstance(obj, nn.Linear) or hasattr(obj, 'weight') or hasattr(obj, 'forward')):
            return obj, '.'.join(path)
    # Brute-force search — accept any module with 'gate' or 'router' in name
    for name, mod in layer.named_modules():
        if ('gate' in name.lower() or 'router' in name.lower()) and hasattr(mod, 'forward'):
            return mod, name
    return None, None

# Identify which layers are MoE
moe_layers = []
for idx in range(N):
    gate, path = find_gate_module(inner.layers[idx])
    if gate is not None:
        moe_layers.append(idx)

print(f'MoE layers: {len(moe_layers)} / {N}', flush=True)
if len(moe_layers) > 0:
    gate_sample, gate_path = find_gate_module(inner.layers[moe_layers[0]])
    num_experts = gate_sample.weight.shape[0] if gate_sample is not None else 128
    print(f'Gate path: layer.{gate_path}', flush=True)
    print(f'Num experts (from gate weight): {num_experts}', flush=True)
else:
    print('ERROR: No MoE layers found!', flush=True)
    sys.exit(1)

# =========================================================================
# Step 1: Define input types and collect prompts
# =========================================================================
print('\n=== Step 1: Input Type Prompts ===', flush=True)

input_types = {
    'math': [
        'What is 127 * 348?',
        'What is 99999 * 99999?',
        'Calculate 15! / 13!',
        'What is 2^16?',
        'Solve for x: 3x + 7 = 22',
        'What is the sum of all integers from 1 to 100?',
        'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
        'What is the greatest common divisor of 144 and 60?',
    ],
    'reasoning': [
        'If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly?',
        'A is taller than B. B is taller than C. Is A taller than C?',
        'If it rains, the ground gets wet. The ground is wet. Did it rain?',
        'There are 5 houses in a row. The red house is to the left of the green house. Where is the blue house?',
        'If no students are teachers, and some teachers are parents, are any students parents?',
        'Alice is twice as old as Bob. Bob is 3 years older than Carol. If Carol is 5, how old is Alice?',
        'A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?',
        'Three friends share a pizza. Each gets 2 slices. How many slices total?',
    ],
    'knowledge': [
        'What is the capital of France?',
        'Who wrote Romeo and Juliet?',
        'What is the chemical formula for water?',
        'In what year did World War II end?',
        'What is the speed of light in meters per second?',
        'What is the largest planet in our solar system?',
        'Who painted the Mona Lisa?',
        'What is the boiling point of water in Celsius?',
    ],
    'creative': [
        'Write a haiku about autumn.',
        'Describe a sunset over the ocean in one sentence.',
        'What emotion would someone feel after losing a close friend?',
        'How would a parent feel seeing their child graduate?',
        'Imagine you are a tree. Describe your day.',
        'Write a metaphor for loneliness.',
        'What does freedom feel like?',
        'Describe the sound of rain on a tin roof.',
    ],
}

for itype, prompts in input_types.items():
    print(f'  {itype}: {len(prompts)} prompts', flush=True)

# =========================================================================
# Step 2: Collect router statistics per input type per layer
# =========================================================================
print('\n=== Step 2: Collect Router Statistics ===', flush=True)
print('For each input type and each MoE layer, computing:', flush=True)
print('  - Expert utilization entropy', flush=True)
print('  - Expert concentration (Herfindahl index)', flush=True)
print('  - Router confidence (softmax peakedness)', flush=True)

# Data structure: router_stats[input_type][layer_idx] = {entropy, concentration, confidence, top_experts}
router_stats = defaultdict(lambda: defaultdict(lambda: {
    'entropies': [], 'concentrations': [], 'confidences': [],
    'expert_counts': None,  # will be initialized
}))

for itype, prompts in input_types.items():
    print(f'\n  Processing {itype}...', flush=True)

    for prompt_idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)

        # Register hooks on all MoE gates
        gate_logits_captured = {}  # layer_idx -> logits tensor
        hooks = []

        for layer_idx in moe_layers:
            gate_mod, _ = find_gate_module(inner.layers[layer_idx])
            if gate_mod is None:
                continue

            def make_hook(lidx):
                def hook_fn(module, input, output):
                    # Router output may be: tensor, tuple (weights, indices), or other
                    if isinstance(output, tuple):
                        # Take first element (usually routing weights or logits)
                        gate_logits_captured[lidx] = output[0].detach().float().cpu()
                    elif isinstance(output, torch.Tensor):
                        gate_logits_captured[lidx] = output.detach().float().cpu()
                    else:
                        # Try input to the router instead (the hidden states)
                        inp = input[0] if isinstance(input, tuple) else input
                        gate_logits_captured[lidx] = inp.detach().float().cpu()
                return hook_fn
            hooks.append(gate_mod.register_forward_hook(make_hook(layer_idx)))

        # Forward pass (no duplication, just baseline routing)
        with torch.no_grad():
            model(inputs['input_ids'], use_cache=False)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Process captured gate logits
        for layer_idx, logits in gate_logits_captured.items():
            # logits shape: [1, seq_len, num_experts]
            probs = torch.softmax(logits, dim=-1)  # [1, seq_len, num_experts]
            probs = probs.squeeze(0)  # [seq_len, num_experts]

            # 1. Expert utilization entropy (per token, then average)
            # Higher entropy = more uniform expert usage = more experts active
            eps = 1e-10
            token_entropies = -(probs * torch.log(probs + eps)).sum(dim=-1)  # [seq_len]
            avg_entropy = token_entropies.mean().item()

            # 2. Expert concentration (Herfindahl index on expert selection counts)
            # Which experts are selected most often across all tokens?
            topk_indices = torch.topk(probs, k=8, dim=-1).indices  # [seq_len, 8]
            expert_counts = torch.zeros(num_experts)
            for tok_experts in topk_indices:
                for e in tok_experts:
                    expert_counts[e.item()] += 1
            # Normalize to frequency
            expert_freq = expert_counts / expert_counts.sum()
            herfindahl = (expert_freq ** 2).sum().item()  # Higher = more concentrated

            # 3. Router confidence (average max probability)
            max_probs = probs.max(dim=-1).values  # [seq_len]
            avg_confidence = max_probs.mean().item()

            # Store
            stats = router_stats[itype][layer_idx]
            stats['entropies'].append(avg_entropy)
            stats['concentrations'].append(herfindahl)
            stats['confidences'].append(avg_confidence)

            # Accumulate expert counts across prompts
            if stats['expert_counts'] is None:
                stats['expert_counts'] = expert_counts.clone()
            else:
                stats['expert_counts'] += expert_counts

        if (prompt_idx + 1) % 4 == 0:
            print(f'    [{prompt_idx+1}/{len(prompts)}] done', flush=True)

# =========================================================================
# Step 3: Analyze routing patterns per input type
# =========================================================================
print('\n=== Step 3: Router Pattern Analysis ===', flush=True)

# Compute per-layer, per-type summary stats
layer_type_summary = {}  # layer_idx -> {type -> {entropy, concentration, confidence, top_5_experts}}

for layer_idx in moe_layers:
    layer_type_summary[layer_idx] = {}
    for itype in input_types:
        stats = router_stats[itype][layer_idx]
        if not stats['entropies']:
            continue
        summary = {
            'entropy': float(np.mean(stats['entropies'])),
            'concentration': float(np.mean(stats['concentrations'])),
            'confidence': float(np.mean(stats['confidences'])),
        }
        if stats['expert_counts'] is not None:
            top5 = torch.topk(stats['expert_counts'], k=5).indices.tolist()
            summary['top_5_experts'] = top5
        layer_type_summary[layer_idx][itype] = summary

# Print summary table (every 4th MoE layer for readability)
print(f'\n{\"Layer\":>5s} | {\"Type\":>10s} | {\"Entropy\":>8s} | {\"Concent\":>8s} | {\"Confid\":>8s} | Top-5 Experts', flush=True)
print('-' * 85, flush=True)
for layer_idx in moe_layers[::4]:  # Every 4th
    for itype in ['math', 'reasoning', 'knowledge', 'creative']:
        if itype in layer_type_summary.get(layer_idx, {}):
            s = layer_type_summary[layer_idx][itype]
            experts_str = str(s.get('top_5_experts', []))
            print(f'{layer_idx:5d} | {itype:>10s} | {s[\"entropy\"]:8.4f} | {s[\"concentration\"]:8.6f} | {s[\"confidence\"]:8.6f} | {experts_str}', flush=True)
    print('-' * 85, flush=True)

# =========================================================================
# Step 4: Compute math-specific expert divergence per layer
# =========================================================================
print('\n=== Step 4: Math-Specific Expert Divergence ===', flush=True)
print('Hypothesis: layers where math routes to DIFFERENT experts than others', flush=True)
print('are the best candidates for math-specific duplication.', flush=True)

math_divergence_scores = {}  # layer_idx -> divergence score

for layer_idx in moe_layers:
    if layer_idx not in layer_type_summary:
        continue
    if 'math' not in layer_type_summary[layer_idx]:
        continue

    math_stats = router_stats['math'][layer_idx]
    if math_stats['expert_counts'] is None:
        continue

    # Math expert frequency distribution
    math_freq = math_stats['expert_counts'] / math_stats['expert_counts'].sum()

    # Average of all other types
    other_freqs = []
    for itype in ['reasoning', 'knowledge', 'creative']:
        other_stats = router_stats[itype][layer_idx]
        if other_stats['expert_counts'] is not None:
            other_freq = other_stats['expert_counts'] / other_stats['expert_counts'].sum()
            other_freqs.append(other_freq)

    if not other_freqs:
        continue

    avg_other_freq = torch.stack(other_freqs).mean(dim=0)

    # Jensen-Shannon divergence between math and average-other
    m = 0.5 * (math_freq + avg_other_freq)
    eps = 1e-10
    kl_math_m = (math_freq * torch.log((math_freq + eps) / (m + eps))).sum().item()
    kl_other_m = (avg_other_freq * torch.log((avg_other_freq + eps) / (m + eps))).sum().item()
    jsd = 0.5 * (kl_math_m + kl_other_m)

    math_divergence_scores[layer_idx] = jsd

    # Also compute: math entropy vs other entropy
    math_entropy = layer_type_summary[layer_idx]['math']['entropy']
    other_entropies = [layer_type_summary[layer_idx][t]['entropy']
                       for t in ['reasoning', 'knowledge', 'creative']
                       if t in layer_type_summary[layer_idx]]
    avg_other_entropy = np.mean(other_entropies) if other_entropies else math_entropy
    math_divergence_scores[layer_idx] = {
        'jsd': jsd,
        'math_entropy': math_entropy,
        'other_entropy': avg_other_entropy,
        'entropy_diff': math_entropy - avg_other_entropy,
    }

# Rank layers by JSD (highest = most math-specific routing)
sorted_by_jsd = sorted(math_divergence_scores.items(),
                        key=lambda x: x[1]['jsd'], reverse=True)

print(f'\nTop 15 layers by math-specific expert divergence (JSD):', flush=True)
for layer_idx, scores in sorted_by_jsd[:15]:
    print(f'  Layer {layer_idx:3d}: JSD={scores[\"jsd\"]:.6f} '
          f'math_entropy={scores[\"math_entropy\"]:.4f} '
          f'other_entropy={scores[\"other_entropy\"]:.4f} '
          f'diff={scores[\"entropy_diff\"]:+.4f}', flush=True)

# =========================================================================
# Step 5: Spectral screen for comparison
# =========================================================================
print('\n=== Step 5: Spectral Screen (for comparison) ===', flush=True)

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
            out_base = model(inputs['input_ids'], use_cache=False)
            logits_base = out_base.logits[:, -1, :].float()

            order = list(range(j)) + list(range(i, j)) + list(range(j, N))
            inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
            set_num_layers(len(order))

            out_dup = model(inputs['input_ids'], use_cache=False)
            logits_dup = out_dup.logits[:, -1, :].float()

            inner.layers = nn.ModuleList(original_layers)
            set_num_layers(N)

            num = torch.norm(logits_dup - logits_base).item()
            den = torch.norm(logits_base).item()
            if den > 1e-8:
                rhos.append(num / den)
    return float(np.mean(rhos)) if rhos else 1.0

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

# Screen single-layer blocks at router-guided candidate layers
# Use the top-10 JSD layers as single-layer candidates
router_guided_candidates = [(layer_idx, layer_idx + 1) for layer_idx, _ in sorted_by_jsd[:10]]
# Also add some multi-layer blocks around top JSD layers
for layer_idx, _ in sorted_by_jsd[:5]:
    for size in [3, 5]:
        start = max(0, layer_idx - size // 2)
        end = min(N, start + size)
        router_guided_candidates.append((start, end))

# Deduplicate
router_guided_candidates = list(set(router_guided_candidates))
router_guided_candidates.sort()

# Also screen the same candidates from standard spectral approach (step=4, sizes 1,3,5,7)
spectral_candidates = []
for start in range(0, N - 1, 4):
    for size in [1, 3, 5, 7]:
        end = start + size
        if end <= N:
            spectral_candidates.append((start, end))

print(f'Screening {len(spectral_candidates)} spectral candidates...', flush=True)
spectral_rhos = {}
for idx, block in enumerate(spectral_candidates):
    spectral_rhos[block] = compute_rho(block)
    if (idx + 1) % 10 == 0:
        print(f'  [{idx+1}/{len(spectral_candidates)}]', flush=True)

sorted_spectral = sorted(spectral_rhos.items(), key=lambda x: x[1])

print(f'\nScreening {len(router_guided_candidates)} router-guided candidates...', flush=True)
router_rhos = {}
for block in router_guided_candidates:
    router_rhos[block] = compute_rho(block)

sorted_router = sorted(router_rhos.items(), key=lambda x: x[1])

print(f'\nSpectral top-5:', flush=True)
for b, r in sorted_spectral[:5]:
    print(f'  ({b[0]:2d},{b[1]:2d}) rho={r:.4f}', flush=True)

print(f'\nRouter-guided top-5:', flush=True)
for b, r in sorted_router[:5]:
    print(f'  ({b[0]:2d},{b[1]:2d}) rho={r:.4f}', flush=True)

# =========================================================================
# Step 6: Evaluate router-guided vs spectral selections
# =========================================================================
print('\n=== Step 6: Evaluation — Router-Guided vs Spectral ===', flush=True)

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

# Baseline
math_base = run_math_probe(gen, verbose=False)
eq_base = run_eq_bench_probe(gen_long, verbose=False)
combined_base = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={combined_base:.2f}', flush=True)

# Evaluate top-5 from each selection method
print(f'\n--- Spectral top-5 evaluation ---', flush=True)
spectral_results = []
for block, rho in sorted_spectral[:5]:
    i, j = block
    order = build_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    delta = combined - combined_base
    print(f'  ({i:2d},{j:2d}) rho={rho:.4f}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} delta={delta:+.2f}', flush=True)
    spectral_results.append({
        'block': list(block), 'rho': rho, 'method': 'spectral',
        'math': math_r['score'], 'eq': eq_r['score'],
        'combined': combined, 'delta': delta,
    })

    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

print(f'\n--- Router-guided top-5 evaluation ---', flush=True)
router_results = []
for block, rho in sorted_router[:5]:
    i, j = block
    order = build_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    delta = combined - combined_base
    print(f'  ({i:2d},{j:2d}) rho={rho:.4f}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} delta={delta:+.2f}', flush=True)
    router_results.append({
        'block': list(block), 'rho': rho, 'method': 'router_guided',
        'math': math_r['score'], 'eq': eq_r['score'],
        'combined': combined, 'delta': delta,
    })

    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

# =========================================================================
# Step 7: Math-only vs non-math evaluation for top router-guided candidates
# =========================================================================
print('\n=== Step 7: Math vs Non-Math Targeted Evaluation ===', flush=True)
print('Testing if router-guided selection specifically helps math.', flush=True)

# Use top-3 router-guided + top-3 spectral blocks (skip duplicates)
all_tested_blocks = []
seen = set()
for block, _ in sorted_router[:3]:
    if block not in seen:
        all_tested_blocks.append(('router', block))
        seen.add(block)
for block, _ in sorted_spectral[:3]:
    if block not in seen:
        all_tested_blocks.append(('spectral', block))
        seen.add(block)

# Additional math prompts for targeted eval
extra_math_prompts = [
    'What is 314159 * 271828?',
    'What is 2 raised to the power of 20?',
    'What is the square root of 152399025?',
    'What is 7777777 * 3333333?',
    'Solve: 15x - 7 = 68',
]

# Non-math quality prompts
non_math_prompts = [
    'Explain photosynthesis in one sentence.',
    'What causes earthquakes?',
    'Describe the feeling of nostalgia.',
    'What is the difference between weather and climate?',
    'Why is the sky blue?',
]

def math_targeted_score(gen_fn):
    \"\"\"Quick math score on extra prompts (just check if answers are reasonable).\"\"\"
    correct = 0
    total = len(extra_math_prompts)
    for prompt in extra_math_prompts:
        full_prompt = f'System: Answer with ONLY the number.\\n\\nUser: {prompt}\\n\\nAssistant:'
        response = gen_fn(full_prompt)
        # Check if response contains a number
        import re
        nums = re.findall(r'\\d+', response.replace(',', ''))
        if nums:
            correct += 1  # At least produced a number
    return correct / total

def non_math_quality_score(gen_fn):
    \"\"\"Quick non-math quality check (response length and coherence proxy).\"\"\"
    scores = []
    for prompt in non_math_prompts:
        response = gen_fn(prompt)
        # Heuristic: longer responses with words = more coherent
        words = response.split()
        score = min(1.0, len(words) / 20.0)  # Normalize: 20+ words = 1.0
        scores.append(score)
    return float(np.mean(scores))

# Baseline targeted scores
print(f'\\nBaseline targeted scores:', flush=True)
base_math_t = math_targeted_score(gen)
base_nonmath_t = non_math_quality_score(gen)
print(f'  Math targeted: {base_math_t:.4f}', flush=True)
print(f'  Non-math quality: {base_nonmath_t:.4f}', flush=True)

targeted_results = []
for method, block in all_tested_blocks:
    i, j = block
    order = build_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

    math_t = math_targeted_score(gen)
    nonmath_t = non_math_quality_score(gen)
    math_delta = math_t - base_math_t
    nonmath_delta = nonmath_t - base_nonmath_t

    print(f'  [{method:>7s}] ({i:2d},{j:2d}): math_targeted={math_t:.4f} (d={math_delta:+.4f}) nonmath={nonmath_t:.4f} (d={nonmath_delta:+.4f})', flush=True)
    targeted_results.append({
        'method': method, 'block': list(block),
        'math_targeted': math_t, 'math_targeted_delta': math_delta,
        'nonmath_quality': nonmath_t, 'nonmath_quality_delta': nonmath_delta,
    })

    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

# =========================================================================
# Summary
# =========================================================================
print('\n' + '=' * 70, flush=True)
print('SUMMARY: MoE Adaptive Routing on Qwen3-30B-A3B', flush=True)
print('=' * 70, flush=True)
print(f'Model: Qwen3-30B-A3B ({N} layers, {len(moe_layers)} MoE)', flush=True)
print(f'Baseline: math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={combined_base:.2f}', flush=True)

# Best from each method
best_spectral = max(spectral_results, key=lambda x: x['combined']) if spectral_results else None
best_router = max(router_results, key=lambda x: x['combined']) if router_results else None

if best_spectral:
    print(f'\nBest SPECTRAL: ({best_spectral[\"block\"][0]},{best_spectral[\"block\"][1]}) '
          f'combined={best_spectral[\"combined\"]:.2f} delta={best_spectral[\"delta\"]:+.2f}', flush=True)
if best_router:
    print(f'Best ROUTER-GUIDED: ({best_router[\"block\"][0]},{best_router[\"block\"][1]}) '
          f'combined={best_router[\"combined\"]:.2f} delta={best_router[\"delta\"]:+.2f}', flush=True)

# Do they agree?
if best_spectral and best_router:
    if best_spectral['block'] == best_router['block']:
        print('\n=> AGREEMENT: spectral and router-guided selected the SAME block!', flush=True)
    else:
        print(f'\n=> DISAGREEMENT: different blocks selected.', flush=True)
        if best_router['combined'] > best_spectral['combined']:
            print('=> Router-guided selection WINS!', flush=True)
        elif best_spectral['combined'] > best_router['combined']:
            print('=> Spectral selection WINS!', flush=True)
        else:
            print('=> Tie.', flush=True)

# Overlap analysis
spectral_set = set(tuple(r['block']) for r in spectral_results)
router_set = set(tuple(r['block']) for r in router_results)
overlap = spectral_set & router_set
print(f'\nBlock overlap: {len(overlap)} / spectral={len(spectral_set)} router={len(router_set)}', flush=True)
if overlap:
    print(f'Shared blocks: {[list(b) for b in overlap]}', flush=True)

# Math-specific routing analysis
print(f'\nMath-specific expert divergence (top-5 layers):', flush=True)
for layer_idx, scores in sorted_by_jsd[:5]:
    print(f'  Layer {layer_idx:3d}: JSD={scores[\"jsd\"]:.6f} entropy_diff={scores[\"entropy_diff\"]:+.4f}', flush=True)

# Targeted eval comparison
print(f'\nTargeted evaluation:', flush=True)
print(f'  Baseline: math_targeted={base_math_t:.4f} nonmath={base_nonmath_t:.4f}', flush=True)
for r in targeted_results:
    print(f'  [{r[\"method\"]:>7s}] ({r[\"block\"][0]},{r[\"block\"][1]}): '
          f'math={r[\"math_targeted\"]:.4f}({r[\"math_targeted_delta\"]:+.4f}) '
          f'nonmath={r[\"nonmath_quality\"]:.4f}({r[\"nonmath_quality_delta\"]:+.4f})', flush=True)

# Save results
os.makedirs('results/data/moe', exist_ok=True)
output = {
    'model': 'Qwen3-30B-A3B',
    'date': datetime.now().isoformat(),
    'num_layers': N,
    'moe_layers': moe_layers,
    'num_experts': num_experts,
    'baseline': {'math': math_base['score'], 'eq': eq_base['score'], 'combined': combined_base},
    'math_divergence_ranking': [
        {'layer': layer_idx, **scores}
        for layer_idx, scores in sorted_by_jsd
    ],
    'spectral_screen': [{'block': list(b), 'rho': r} for b, r in sorted_spectral],
    'router_guided_screen': [{'block': list(b), 'rho': r} for b, r in sorted_router],
    'spectral_eval': spectral_results,
    'router_guided_eval': router_results,
    'targeted_eval': {
        'baseline_math': base_math_t,
        'baseline_nonmath': base_nonmath_t,
        'results': targeted_results,
    },
    'layer_type_summary': {
        str(k): {itype: stats for itype, stats in v.items()}
        for k, v in layer_type_summary.items()
    },
}

with open('results/data/moe/adaptive_routing.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f'\nSaved to results/data/moe/adaptive_routing.json', flush=True)
"

echo "=== Done at $(date) ==="
