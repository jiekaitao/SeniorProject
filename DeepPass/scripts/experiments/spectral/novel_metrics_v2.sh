#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_novel_metrics_v2_%j.log
#SBATCH --job-name=deeppass_nmv2

# Novel screening metrics v2 — fixed tensor size handling
# Tests OTAS, GCHS, SCPAM, CLRG against 25-block validation set

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Novel Screening Metrics v2 ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
# Use fixed-length prompts to avoid tensor size mismatches
CAL_PROMPTS = [
    'What is 127 * 348?', 'What is 99999 * 99999?',
    'Calculate 15! / 13!', 'What is 2^16?',
    'What is the sum of all integers from 1 to 100?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'What emotion would someone feel after losing a close friend?',
    'How would a parent feel seeing their child graduate?',
]
MAX_LEN = 32  # Fixed token length for all prompts

TEST_BLOCKS = [
    (0, 3), (0, 7), (4, 9), (5, 12), (8, 13), (10, 17), (15, 20),
    (20, 27), (22, 27), (25, 32), (28, 33), (30, 37), (35, 40),
    (40, 45), (40, 47), (42, 49), (45, 50), (45, 52),
    (50, 55), (50, 60), (55, 60), (55, 62), (60, 65), (65, 72), (70, 77),
]

print('=' * 70)
print('NOVEL SCREENING METRICS v2')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

# Load ground truth
with open('results/data/72b/fresh_validation/results.json') as f:
    fresh = json.load(f)
ground_truth = {tuple(r['block']): r for r in fresh['results']}

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
# Collect all hidden states with FIXED token length
# =====================================================================
print('\\n--- Collecting base hidden states (fixed length) ---', flush=True)

base_states = {}  # layer_idx -> [n_prompts, T, d] tensor

for prompt_idx, prompt in enumerate(CAL_PROMPTS):
    ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_LEN,
                    padding='max_length').to(device)
    hooks = []
    layer_outs = {}

    def make_hook(idx):
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            layer_outs[idx] = out.detach().float().cpu().squeeze(0)  # [T, d]
        return hook_fn

    for idx in range(N):
        hooks.append(original_layers[idx].register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        model(ids['input_ids'], use_cache=False)

    for h in hooks:
        h.remove()

    for idx, tensor in layer_outs.items():
        if idx not in base_states:
            base_states[idx] = []
        base_states[idx].append(tensor)

# Stack into tensors
for idx in base_states:
    base_states[idx] = torch.stack(base_states[idx])  # [n_prompts, T, d]

print(f'  Collected {len(base_states)} layers, shape per layer: {base_states[0].shape}', flush=True)

# =====================================================================
# Collect duplicated exit states
# =====================================================================
print('--- Collecting duplicated exit states ---', flush=True)

dup_exit = {}  # block -> [n_prompts, T, d]

for bi, block in enumerate(TEST_BLOCKS):
    i, j = block
    order = build_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)

    # Hook the last layer of the block — it fires twice (first+second pass)
    # We only want the second firing per prompt
    target_pos = j + (j - i) - 1
    captured = []
    fire_count = [0]

    def cap_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        fire_count[0] += 1
        if fire_count[0] % 2 == 0:  # only capture second firing (second pass)
            captured.append(out.detach().float().cpu().squeeze(0))

    hh = inner.layers[target_pos].register_forward_hook(cap_hook)

    for prompt in CAL_PROMPTS:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_LEN,
                        padding='max_length').to(device)
        with torch.no_grad():
            model(ids['input_ids'], use_cache=False)

    hh.remove()
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    dup_exit[block] = torch.stack(captured)  # [n_prompts, T, d]
    if (bi + 1) % 5 == 0:
        print(f'  [{bi+1}/{len(TEST_BLOCKS)}]', flush=True)

print(f'  Done. Shape: {dup_exit[TEST_BLOCKS[0]].shape}', flush=True)

# =====================================================================
# Helper functions
# =====================================================================

def cos_sim(a, b):
    \"\"\"Cosine similarity between two [T, d] tensors, averaged over tokens.\"\"\"
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # [T]
    return float(cos.mean())

def cos_sim_batch(a, b):
    \"\"\"Cosine sim between [n, T, d] tensors, averaged over prompts and tokens.\"\"\"
    # a, b same shape
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # [n, T]
    return float(cos.mean())

# =====================================================================
# METRIC 1: OTAS (On-Trajectory Advancement Score)
# =====================================================================
print('\\n--- OTAS ---', flush=True)
K = 8
otas_scores = {}

for block in TEST_BLOCKS:
    i, j = block
    h_dup = dup_exit[block]  # [n, T, d]
    h_j = base_states.get(j - 1)  # [n, T, d]
    if h_j is None or j + K - 1 >= N:
        otas_scores[block] = 0.0
        continue

    current_sim = cos_sim_batch(h_dup, h_j)
    future_sims = []
    for k in range(1, K + 1):
        if j + k - 1 in base_states:
            future_sims.append(cos_sim_batch(h_dup, base_states[j + k - 1]))

    otas_scores[block] = (max(future_sims) - current_sim) if future_sims else 0.0

# =====================================================================
# METRIC 2: GCHS (Geodesic Curvature Hotspot Score)
# =====================================================================
print('--- GCHS ---', flush=True)
gchs_scores = {}

for block in TEST_BLOCKS:
    i, j = block
    curvatures = []
    for l in range(i, j - 1):
        if l in base_states and l + 1 in base_states:
            v1 = base_states[l + 1] - base_states[l]  # [n, T, d]
            if l > 0 and l - 1 in base_states:
                v0 = base_states[l] - base_states[l - 1]
            else:
                v0 = v1
            # Cosine between consecutive velocity vectors, averaged
            cos_val = cos_sim_batch(v0, v1)
            mag = 0.5 * (v0.norm(dim=-1).mean().item() + v1.norm(dim=-1).mean().item())
            curvatures.append((1.0 - cos_val) * mag)
    gchs_scores[block] = float(np.mean(curvatures)) if curvatures else 0.0

# =====================================================================
# METRIC 3: CLRG (Cross-Layer Redundancy Gap)
# =====================================================================
print('--- CLRG ---', flush=True)
clrg_scores = {}

for block in TEST_BLOCKS:
    i, j = block
    L = j - i
    # Block residual
    h_j = base_states.get(j - 1)
    h_i = base_states.get(i - 1) if i > 0 else base_states.get(0)  # approximate
    if h_j is None or h_i is None:
        clrg_scores[block] = 0.0
        continue
    block_resid = h_j - h_i  # [n, T, d]

    # Future similarity
    future_sims = []
    for k in range(1, 5):
        fi, fj = j, min(j + L, N)
        if fj - 1 in base_states and fi - 1 in base_states:
            future_resid = base_states[fj - 1] - base_states[fi - 1]
            future_sims.append(cos_sim_batch(block_resid, future_resid))
        fi, fj = j, j + k
        if fj - 1 < N and fj - 1 in base_states:
            break

    # Past similarity
    past_sims = []
    for k in range(1, 5):
        pi, pj = max(0, i - L), i
        if pj - 1 in base_states and (pi - 1 in base_states or pi == 0):
            pi_state = base_states.get(pi - 1, base_states.get(0))
            past_resid = base_states[pj - 1] - pi_state
            past_sims.append(cos_sim_batch(block_resid, past_resid))
        break

    f = max(future_sims) if future_sims else 0
    p = max(past_sims) if past_sims else 0
    clrg_scores[block] = f - p

# =====================================================================
# METRIC 4: SCPAM (simplified — alignment of dup exit with future layers)
# =====================================================================
print('--- SCPAM (simplified) ---', flush=True)
scpam_scores = {}

for block in TEST_BLOCKS:
    i, j = block
    h_dup = dup_exit[block]
    # How well does the dup output align with base states at j, j+1, ..., j+K?
    alignments = []
    for k in range(0, min(8, N - j)):
        if j + k - 1 in base_states:
            alignments.append(cos_sim_batch(h_dup, base_states[j + k - 1]))
    # Best alignment minus alignment with input (j-1)
    if alignments and j - 1 in base_states:
        input_align = cos_sim_batch(h_dup, base_states[j - 1])
        scpam_scores[block] = max(alignments) - input_align
    else:
        scpam_scores[block] = 0.0

# =====================================================================
# CORRELATION ANALYSIS
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('CORRELATION ANALYSIS')
print(f'{\"=\" * 70}', flush=True)

deltas = [ground_truth[block]['delta'] for block in TEST_BLOCKS]

metrics = {
    'otas': [otas_scores[b] for b in TEST_BLOCKS],
    'gchs': [gchs_scores[b] for b in TEST_BLOCKS],
    'clrg': [clrg_scores[b] for b in TEST_BLOCKS],
    'scpam': [scpam_scores[b] for b in TEST_BLOCKS],
    'rho': [ground_truth[b]['rho'] for b in TEST_BLOCKS],
    'blood': [ground_truth[b]['blood_impact'] for b in TEST_BLOCKS],
    'sbuid_6k': [ground_truth[b]['blood_impact'] - 6000 * ground_truth[b]['rho'] for b in TEST_BLOCKS],
}

print(f'{\"Metric\":>15s} {\"Spearman r\":>12s} {\"p-value\":>10s} {\"Pearson r\":>12s} {\"p-value\":>10s} {\"Top5\":>6s}')
print('-' * 70)

for name, scores in metrics.items():
    if all(s == 0 for s in scores):
        print(f'{name:>15s}   ALL ZERO')
        continue
    rs, ps = spearmanr(scores, deltas)
    rp, pp = pearsonr(scores, deltas)
    top5_pred = set(np.argsort([-s for s in scores])[:5])
    top5_true = set(np.argsort([-d for d in deltas])[:5])
    overlap = len(top5_pred & top5_true)
    sig = '***' if ps < 0.01 else '**' if ps < 0.05 else '*' if ps < 0.1 else ''
    print(f'{name:>15s} {rs:+12.3f} {ps:10.4f} {rp:+12.3f} {pp:10.4f} {overlap:4d}/5 {sig}')

# Combined metrics
print(f'\\n--- Combinations ---')
combos = {
    'otas+blood': lambda i: metrics['otas'][i] + 0.001 * metrics['blood'][i],
    'gchs+sbuid': lambda i: metrics['gchs'][i] + 0.001 * metrics['sbuid_6k'][i],
    'otas+sbuid': lambda i: metrics['otas'][i] + 0.001 * metrics['sbuid_6k'][i],
    'otas+gchs': lambda i: metrics['otas'][i] + 0.5 * metrics['gchs'][i],
    'gchs*blood': lambda i: metrics['gchs'][i] * max(0, metrics['blood'][i]),
}
for name, fn in combos.items():
    scores = [fn(i) for i in range(len(TEST_BLOCKS))]
    rs, ps = spearmanr(scores, deltas)
    top5_pred = set(np.argsort([-s for s in scores])[:5])
    top5_true = set(np.argsort([-d for d in deltas])[:5])
    overlap = len(top5_pred & top5_true)
    sig = '***' if ps < 0.01 else '**' if ps < 0.05 else '*' if ps < 0.1 else ''
    print(f'{name:>15s} {rs:+12.3f} {ps:10.4f} {overlap:4d}/5 {sig}')

# Per-block details
print(f'\\n--- Per-block scores ---')
print(f'{\"Block\":>8s} {\"delta\":>7s} {\"otas\":>8s} {\"gchs\":>8s} {\"clrg\":>8s} {\"scpam\":>8s} {\"sbuid\":>8s}')
for i, block in enumerate(sorted(TEST_BLOCKS, key=lambda b: ground_truth[b]['delta'], reverse=True)):
    d = ground_truth[block]['delta']
    print(f'({block[0]:2d},{block[1]:2d}) {d:+7.2f} {otas_scores[block]:+8.4f} {gchs_scores[block]:8.2f} '
          f'{clrg_scores[block]:+8.4f} {scpam_scores[block]:+8.4f} '
          f'{ground_truth[block][\"blood_impact\"] - 6000*ground_truth[block][\"rho\"]:+8.0f}', flush=True)

# Save
os.makedirs('results/data/72b/novel_metrics', exist_ok=True)
output = {
    'date': datetime.now().isoformat(),
    'n_blocks': len(TEST_BLOCKS),
    'metrics': {str(b): {
        'otas': otas_scores[b], 'gchs': gchs_scores[b],
        'clrg': clrg_scores[b], 'scpam': scpam_scores[b],
    } for b in TEST_BLOCKS},
}
with open('results/data/72b/novel_metrics/v2_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f'\\nSaved to results/data/72b/novel_metrics/v2_results.json', flush=True)
"

echo "=== Done at $(date) ==="
