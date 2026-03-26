#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_novel_metrics_%j.log
#SBATCH --job-name=deeppass_nmet

# Implement and validate top screening metrics from GPT-5.4 Pro analysis:
# 1. OTAS (On-Trajectory Advancement Score) — cheapest serious metric
# 2. GCHS (Geodesic Curvature Hotspot Score) — cheap region detector
# 3. FGAS (Fisher-Geodesic Advancement Score) — Fisher-weighted version
# 4. FFAS (Fisher Fixed-Point Anisotropy Score) — denoiser hypothesis test
# 5. SCPAM (Shifted Cross-Pass Alignment Matrix) — internal dynamics
# All validated against our 25-block fresh dataset

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Novel Screening Metrics ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('=' * 70)
print('NOVEL SCREENING METRICS — GPT-5.4 Pro Analysis')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers', flush=True)

CAL_PROMPTS = [
    'What is 127 * 348?', 'What is 99999 * 99999?',
    'Calculate 15! / 13!', 'What is 2^16?',
    'What is the sum of all integers from 1 to 100?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'What emotion would someone feel after losing a close friend?',
    'How would a parent feel seeing their child graduate?',
    'Describe the feeling of watching a sunset after a difficult day.',
    'What is the derivative of sin(x) * e^x?',
    'def fibonacci(n): # complete this function',
    'Explain the concept of entropy in thermodynamics.',
    'Three friends split a dollar 147 bill with 20% tip. How much each?',
    'What is 13^3?',
    'The theory of general relativity describes',
    'How does anticipation differ from anxiety?',
]

# Test blocks — same as fresh validation
TEST_BLOCKS = [
    (0, 3), (0, 7), (4, 9), (5, 12), (8, 13), (10, 17), (15, 20),
    (20, 27), (22, 27), (25, 32), (28, 33), (30, 37), (35, 40),
    (40, 45), (40, 47), (42, 49), (45, 50), (45, 52),
    (50, 55), (50, 60), (55, 60), (55, 62), (60, 65), (65, 72), (70, 77),
]

# Load ground truth deltas
with open('results/data/72b/fresh_validation/results.json') as f:
    fresh = json.load(f)
ground_truth = {tuple(r['block']): r['delta'] for r in fresh['results']}

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

# =====================================================================
# Step 1: Collect base hidden states for all layers on calibration prompts
# =====================================================================
print('\\n--- Collecting base hidden states ---', flush=True)

base_states = {}  # layer_idx -> list of [T, d] tensors (one per prompt)
hooks = []
layer_outputs = {}

def make_hook(idx):
    def hook_fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if idx not in layer_outputs:
            layer_outputs[idx] = []
        layer_outputs[idx].append(out.detach().float().cpu())
    return hook_fn

for idx in range(N):
    hooks.append(original_layers[idx].register_forward_hook(make_hook(idx)))

# Also capture embedding output
embed_outputs = []
def embed_hook(module, input, output):
    embed_outputs.append(output.detach().float().cpu())
embed_h = inner.embed_tokens.register_forward_hook(embed_hook)

for prompt in CAL_PROMPTS:
    ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).to(device)
    with torch.no_grad():
        model(ids['input_ids'], use_cache=False)

for h in hooks:
    h.remove()
embed_h.remove()

base_states = layer_outputs  # layer_idx -> list of tensors
print(f'  Collected {len(base_states)} layers x {len(CAL_PROMPTS)} prompts', flush=True)

# =====================================================================
# Step 2: Collect duplicated hidden states at exit for each block
# =====================================================================
print('\\n--- Collecting duplicated states ---', flush=True)

dup_exit_states = {}  # block -> list of [T, d] tensors

for block_idx, block in enumerate(TEST_BLOCKS):
    i, j = block
    dup_exits = []

    # Hook the last layer of the block during the SECOND pass
    order = build_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx2] for idx2 in order])
    model.config.num_hidden_layers = len(order)

    # The duplicated exit is at position j + (j-i) in the extended order
    # But easier: just hook the layer at position j-1 and capture second occurrence
    capture = {'count': 0, 'states': []}
    last_layer_idx = j - 1

    def exit_hook(module, input, output, target_count=2):
        out = output[0] if isinstance(output, tuple) else output
        capture['count'] += 1
        if capture['count'] == target_count:
            capture['states'].append(out.detach().float().cpu())

    # Actually, since we swapped layers, the second occurrence of layer j-1
    # in the order is at position j + (j-i) - 1. Let's use a different approach:
    # just run the full forward and capture the hidden state after the duplicated block exits.
    # That's the state at position j + (j-i) in the layer_order, which feeds into original layer j.

    # Simplest: capture input to layer j in the duplicated model
    # In the duplicated order, the first occurrence of layer j is at position j + (j-i)
    dup_exit_pos = j + (j - i)  # position in the extended layer list where layer j starts

    # Hook approach: capture the hidden state BEFORE layer j fires (= duplicated block exit)
    captured = []
    def pre_hook_fn(module, input):
        inp = input[0] if isinstance(input, tuple) else input
        captured.append(inp.detach().float().cpu())

    # The layer at position dup_exit_pos in the order is original_layers[order[dup_exit_pos]] = original_layers[j]
    # But multiple positions map to the same layer object, so we need to be careful.
    # Use a counter instead.
    layer_j = original_layers[j] if j < N else None
    if layer_j is not None:
        call_count = [0]
        target_call = 2  # layer j is called twice: once at original position, once after dup block
        # Actually in our order, layer j appears only once (at position j + block_size)
        # Wait, no. The order is: [0..j-1, i..j-1, j..N-1]. Layer j appears once.
        # The duplicated block exit is the output of the second copy of layer j-1.
        # Let me just use hooks on the duplicated model layers.

        # Cleaner: register a hook on the layer at position (j + block_size - 1) in the extended list
        # That's the last layer of the second pass.
        target_pos = j + (j - i) - 1
        hooked_layer = inner.layers[target_pos]
        def cap_hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            captured.append(out.detach().float().cpu())
        hh = hooked_layer.register_forward_hook(cap_hook)

        for prompt in CAL_PROMPTS:
            ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).to(device)
            with torch.no_grad():
                model(ids['input_ids'], use_cache=False)

        hh.remove()
        dup_exit_states[block] = captured

    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    if (block_idx + 1) % 5 == 0:
        print(f'  [{block_idx+1}/{len(TEST_BLOCKS)}] done', flush=True)

print(f'  Collected dup exits for {len(dup_exit_states)} blocks', flush=True)

# =====================================================================
# Step 3: Compute all metrics
# =====================================================================

def cosine_flat(a, b):
    a, b = a.float(), b.float()
    # Truncate to same number of tokens
    T = min(a.shape[0], b.shape[0])
    a_flat = a[:T].reshape(-1)
    b_flat = b[:T].reshape(-1)
    return float(torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)))

def sample_cos(a, b):
    \"\"\"Sample-wise cosine: mean cosine across tokens. Handles different seq lengths.\"\"\"
    a, b = a.float(), b.float()
    # Handle different dimensions — flatten and use global cosine if shapes mismatch
    if a.shape != b.shape:
        return cosine_flat(a, b)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return float(cos.mean())

def linear_cka(a, b):
    \"\"\"Linear CKA between two activation matrices [T, d].\"\"\"
    a, b = a.float(), b.float()
    T = min(a.shape[0], b.shape[0])
    a, b = a[:T], b[:T]
    # Center
    a = a - a.mean(0)
    b = b - b.mean(0)
    # CKA
    ab = torch.trace(a.T @ b @ b.T @ a)
    aa = torch.trace(a.T @ a @ a.T @ a)
    bb = torch.trace(b.T @ b @ b.T @ b)
    return float(ab / (torch.sqrt(aa * bb) + 1e-8))

all_metrics = {block: {} for block in TEST_BLOCKS}

# --- METRIC 1: OTAS (On-Trajectory Advancement Score) ---
print('\\n--- Computing OTAS ---', flush=True)
K_otas = 8

for block in TEST_BLOCKS:
    i, j = block
    if block not in dup_exit_states or j + K_otas >= N:
        all_metrics[block]['otas'] = 0.0
        continue

    scores = []
    for prompt_idx in range(len(CAL_PROMPTS)):
        if prompt_idx >= len(dup_exit_states[block]):
            continue
        h_dup = dup_exit_states[block][prompt_idx]
        h_j = base_states[j-1][prompt_idx] if j-1 in base_states else None
        if h_j is None:
            continue

        # Future similarities
        future_sims = []
        for k in range(1, K_otas + 1):
            if j + k - 1 < N and (j + k - 1) in base_states:
                future_sims.append(sample_cos(h_dup, base_states[j + k - 1][prompt_idx]))

        # Current similarity
        current_sim = sample_cos(h_dup, h_j)

        if future_sims:
            scores.append(max(future_sims) - current_sim)

    all_metrics[block]['otas'] = float(np.mean(scores)) if scores else 0.0

# --- METRIC 2: GCHS (Geodesic Curvature Hotspot Score) ---
print('--- Computing GCHS ---', flush=True)

for block in TEST_BLOCKS:
    i, j = block
    curvatures = []
    for prompt_idx in range(len(CAL_PROMPTS)):
        for l in range(i, j - 1):
            if l in base_states and l + 1 in base_states and l - 1 >= 0 and (l - 1) in base_states:
                # v_l = H_{l+1} - H_l
                v1 = base_states[l][prompt_idx] - (base_states[l-1][prompt_idx] if l > 0 else embed_outputs[prompt_idx])
                v2 = base_states[l+1][prompt_idx] - base_states[l][prompt_idx] if l+1 in base_states else v1
                cos_val = 1.0 - cosine_flat(v1, v2)
                mag = 0.5 * (v1.norm().item() + v2.norm().item())
                curvatures.append(cos_val * mag)
    all_metrics[block]['gchs'] = float(np.mean(curvatures)) if curvatures else 0.0

# --- METRIC 3: SCPAM (Shifted Cross-Pass Alignment) ---
print('--- Computing SCPAM ---', flush=True)

for block in TEST_BLOCKS:
    i, j = block
    L = j - i
    if L < 2:
        all_metrics[block]['scpam'] = 0.0
        continue

    # Collect internal states during first and second pass
    # We need to re-run the duplicated model with hooks on each internal layer
    order = build_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx2] for idx2 in order])
    model.config.num_hidden_layers = len(order)

    pass1_states = {offset: [] for offset in range(L)}
    pass2_states = {offset: [] for offset in range(L)}
    layer_call_counts = {}

    hooks2 = []
    for offset in range(L):
        layer_idx_in_block = i + offset
        # In the order, this layer appears twice: at position (i + offset) and at position (j + offset)
        pos1 = i + offset
        pos2 = j + offset

        def make_pass_hook(off, p1, p2):
            counts = [0]
            def hook_fn(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                counts[0] += 1
                if counts[0] % 2 == 1:  # first pass (odd calls)
                    pass1_states[off].append(out.detach().float().cpu())
                else:  # second pass (even calls)
                    pass2_states[off].append(out.detach().float().cpu())
            return hook_fn

        # The layer object for layer_idx_in_block is shared between pos1 and pos2
        layer_obj = original_layers[layer_idx_in_block]
        hooks2.append(layer_obj.register_forward_hook(make_pass_hook(offset, pos1, pos2)))

    for prompt in CAL_PROMPTS[:8]:  # fewer prompts for speed
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).to(device)
        with torch.no_grad():
            model(ids['input_ids'], use_cache=False)

    for h in hooks2:
        h.remove()
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    # Build similarity matrix and score
    best_band = 0
    n_prompts = min(len(pass1_states[0]), len(pass2_states[0]))
    if n_prompts > 0 and L >= 2:
        for shift in range(min(4, L)):
            band_sims = []
            for m in range(L):
                n = min(L - 1, m + shift)
                for p in range(n_prompts):
                    if m < len(pass2_states) and n < len(pass1_states):
                        if p < len(pass2_states[m]) and p < len(pass1_states[n]):
                            band_sims.append(sample_cos(pass2_states[m][p], pass1_states[n][p]))
            if band_sims:
                best_band = max(best_band, np.mean(band_sims))

    all_metrics[block]['scpam'] = best_band

# --- METRIC 4: CLRG (Cross-Layer Redundancy Gap) ---
print('--- Computing CLRG ---', flush=True)

K_clrg = 8
for block in TEST_BLOCKS:
    i, j = block
    # Block residual: H_j - H_i for each prompt
    block_resids = []
    for p in range(len(CAL_PROMPTS)):
        if j-1 in base_states and (i-1 in base_states or i == 0):
            h_j = base_states[j-1][p]
            h_i = base_states[i-1][p] if i > 0 and i-1 in base_states else embed_outputs[p]
            block_resids.append(h_j - h_i)

    if not block_resids:
        all_metrics[block]['clrg'] = 0.0
        continue

    # Future block residuals
    future_sims = []
    for k in range(1, K_clrg + 1):
        fj = j + k * (j - i)
        if fj - 1 < N and fj - 1 in base_states:
            fi = j + (k - 1) * (j - i)
            fi_key = fi - 1 if fi > 0 else -1
            future_resids = []
            for p in range(len(CAL_PROMPTS)):
                if fj - 1 in base_states and (fi_key in base_states or fi == 0):
                    h_fj = base_states[fj-1][p]
                    h_fi = base_states[fi_key][p] if fi_key >= 0 and fi_key in base_states else embed_outputs[p]
                    future_resids.append(h_fj - h_fi)
            if future_resids and len(future_resids) == len(block_resids):
                # CKA between block_resids and future_resids
                sims = [cosine_flat(a, b) for a, b in zip(block_resids, future_resids)]
                future_sims.append(np.mean(sims))

    # Past block residuals
    past_sims = []
    for k in range(1, K_clrg + 1):
        pi = i - k * (j - i)
        if pi >= 0 and pi - 1 >= -1:
            pj = i - (k - 1) * (j - i)
            past_resids = []
            for p in range(len(CAL_PROMPTS)):
                pi_key = pi - 1 if pi > 0 else -1
                pj_key = pj - 1
                if pj_key in base_states and (pi_key in base_states or pi == 0):
                    h_pj = base_states[pj_key][p]
                    h_pi = base_states[pi_key][p] if pi_key >= 0 and pi_key in base_states else embed_outputs[p]
                    past_resids.append(h_pj - h_pi)
            if past_resids and len(past_resids) == len(block_resids):
                sims = [cosine_flat(a, b) for a, b in zip(block_resids, past_resids)]
                past_sims.append(np.mean(sims))

    future_max = max(future_sims) if future_sims else 0
    past_max = max(past_sims) if past_sims else 0
    all_metrics[block]['clrg'] = future_max - past_max

print('--- All metrics computed ---', flush=True)

# =====================================================================
# Step 4: Correlation analysis
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('CORRELATION ANALYSIS')
print(f'{\"=\" * 70}', flush=True)

deltas = [ground_truth.get(block, 0) for block in TEST_BLOCKS]

# Also load existing rho and blood
existing = {tuple(r['block']): r for r in fresh['results']}

print(f'{\"Metric\":>20s} {\"Spearman r\":>12s} {\"p-value\":>10s} {\"Pearson r\":>12s} {\"p-value\":>10s} {\"Top5\":>6s}')
print('-' * 75)

for metric_name in ['otas', 'gchs', 'scpam', 'clrg']:
    scores = [all_metrics[block].get(metric_name, 0) for block in TEST_BLOCKS]
    if all(s == 0 for s in scores):
        print(f'{metric_name:>20s}: ALL ZERO — skipped')
        continue
    rs, ps = spearmanr(scores, deltas)
    rp, pp = pearsonr(scores, deltas)
    # Top-5 overlap
    top5_pred = set(np.argsort([-s for s in scores])[:5])
    top5_true = set(np.argsort([-d for d in deltas])[:5])
    overlap = len(top5_pred & top5_true)
    sig = '***' if ps < 0.01 else '**' if ps < 0.05 else '*' if ps < 0.1 else ''
    print(f'{metric_name:>20s} {rs:+12.3f} {ps:10.4f} {rp:+12.3f} {pp:10.4f} {overlap:4d}/5 {sig}')

# Also show existing metrics for comparison
for metric_name, getter in [('rho', lambda b: existing[b]['rho']),
                              ('blood', lambda b: existing[b]['blood_impact']),
                              ('sbuid_6k', lambda b: existing[b]['blood_impact'] - 6000 * existing[b]['rho'])]:
    scores = [getter(block) for block in TEST_BLOCKS if block in existing]
    d = [ground_truth.get(block, 0) for block in TEST_BLOCKS if block in existing]
    rs, ps = spearmanr(scores, d)
    rp, pp = pearsonr(scores, d)
    top5_pred = set(np.argsort([-s for s in scores])[:5])
    top5_true = set(np.argsort([-dd for dd in d])[:5])
    overlap = len(top5_pred & top5_true)
    sig = '***' if ps < 0.01 else '**' if ps < 0.05 else '*' if ps < 0.1 else ''
    print(f'{metric_name:>20s} {rs:+12.3f} {ps:10.4f} {rp:+12.3f} {pp:10.4f} {overlap:4d}/5 {sig}')

# Combined metrics
print(f'\\n--- Combined metrics ---')
for combo_name, combo_fn in [
    ('otas+gchs', lambda b: all_metrics[b].get('otas',0) + 0.5*all_metrics[b].get('gchs',0)),
    ('otas+sbuid', lambda b: all_metrics[b].get('otas',0) + 0.001*(existing[b]['blood_impact'] - 6000*existing[b]['rho']) if b in existing else 0),
    ('gchs+sbuid', lambda b: all_metrics[b].get('gchs',0) + 0.001*(existing[b]['blood_impact'] - 6000*existing[b]['rho']) if b in existing else 0),
]:
    scores = [combo_fn(block) for block in TEST_BLOCKS]
    d = [ground_truth.get(block, 0) for block in TEST_BLOCKS]
    rs, ps = spearmanr(scores, d)
    top5_pred = set(np.argsort([-s for s in scores])[:5])
    top5_true = set(np.argsort([-dd for dd in d])[:5])
    overlap = len(top5_pred & top5_true)
    sig = '***' if ps < 0.01 else '**' if ps < 0.05 else '*' if ps < 0.1 else ''
    print(f'{combo_name:>20s} {rs:+12.3f} {ps:10.4f} {overlap:4d}/5 {sig}')

# Save
os.makedirs('results/data/72b/novel_metrics', exist_ok=True)
output = {
    'date': datetime.now().isoformat(),
    'n_blocks': len(TEST_BLOCKS),
    'metrics': {str(k): v for k, v in all_metrics.items()},
    'ground_truth': {str(k): v for k, v in ground_truth.items()},
}
with open('results/data/72b/novel_metrics/results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f'\\nSaved to results/data/72b/novel_metrics/results.json', flush=True)
"

echo "=== Done at $(date) ==="
