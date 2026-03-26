#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ffn_danger_%j.log
#SBATCH --job-name=deeppass_fdang

# FFN Danger Score (72B)
#
# Computes R^FFN per layer in block (45,52) and correlates with actual harm.
# For each layer's FFN:
#   1. Jacobian-vector product J_FFN * delta (amplification of seam perturbation)
#   2. FFN output margin gamma (top-2 gating activation difference)
#   3. Local key-value conflict C_l
#   4. Danger score R^FFN = C_l * |J_FFN * delta| / (gamma + eps)
#   5. Active-set stability (Jaccard overlap of top-k gate activations)
#
# Correlates with known sublayer harm from sublayer_duplication experiment.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== FFN Danger Score ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
BLOCK = (45, 52)
BLOCK_SIZE = BLOCK[1] - BLOCK[0]  # 7 layers

# Calibration prompts
CAL_PROMPTS = [
    'What is 127 * 348?',
    'What is 99999 * 99999?',
    'Calculate 15! / 13!',
    'What is 2^16?',
    'What is the sum of all integers from 1 to 100?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'What emotion would someone feel after losing a close friend?',
    'How would a parent feel seeing their child graduate?',
]

# Known sublayer harm from sublayer duplication experiment
# harm = (score with FFN removed) - (score with full dup)
# Negative means removing FFN HELPED (i.e., FFN was harmful during re-retrieval)
KNOWN_FFN_HARM = {
    0: +0.03,   # L0 (layer 45): FFN neutral
    1: -0.56,   # L1 (layer 46): slight harm from removing FFN
    2: -3.00,   # L2 (layer 47): FFN is VERY harmful
    3: -1.85,   # L3 (layer 48): moderate harm
    4: -3.02,   # L4 (layer 49): very harmful
    5: -3.43,   # L5 (layer 50): most harmful
    6: -2.97,   # L6 (layer 51): very harmful
}

print('=' * 70)
print('FFN DANGER SCORE ANALYSIS')
print('Predicting which FFNs corrupt factual retrieval during re-pass')
print(f'Block: {BLOCK}, Layers: {BLOCK[0]}-{BLOCK[1]-1}')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers on {device}', flush=True)

# =====================================================================
# Build layer order and find seam positions
# =====================================================================

def build_order(block, N):
    i, j = block
    return list(range(j)) + list(range(i, j)) + list(range(j, N))

layer_order = build_order(BLOCK, N)

# =====================================================================
# Collect per-layer seam deltas
# For each layer in the block, we collect:
#   - h_in_first: input to FFN on first pass
#   - h_in_second: input to FFN on second pass
#   - delta = h_in_second - h_in_first (the seam perturbation at FFN input)
# =====================================================================

def collect_per_layer_ffn_inputs(prompt):
    \"\"\"
    Run full duplicated forward, collecting FFN inputs at each layer in the block
    for both first and second pass.
    Returns dict: layer_offset -> (ffn_input_first, ffn_input_second)
    \"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

    # We need to track which step corresponds to which pass
    # In layer_order, layers BLOCK[0]..BLOCK[1]-1 appear twice
    block_layers = set(range(BLOCK[0], BLOCK[1]))
    pass_count = {}  # layer_idx -> count of times we've seen it
    ffn_inputs = {}  # (layer_idx, pass_num) -> FFN input tensor

    with torch.no_grad():
        h = inner.embed_tokens(input_ids)
        seq_len = h.shape[1]
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = inner.rotary_emb(h, pos_ids)

        for step_idx, layer_idx in enumerate(layer_order):
            layer = original_layers[layer_idx]

            if layer_idx in block_layers:
                pass_count[layer_idx] = pass_count.get(layer_idx, 0) + 1
                pass_num = pass_count[layer_idx]

                # Manual sublayer forward to capture FFN input
                residual = h
                normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(normed, position_embeddings=pos_embeds, attention_mask=None, use_cache=False)
                attn_out = attn_out[0] if isinstance(attn_out, tuple) else attn_out
                h = residual + attn_out

                # h is now the input to post_attention_layernorm -> FFN
                ffn_input_normed = layer.post_attention_layernorm(h)
                ffn_inputs[(layer_idx, pass_num)] = ffn_input_normed.clone()

                # Run FFN
                residual = h
                ffn_out = layer.mlp(ffn_input_normed)
                h = residual + ffn_out
            else:
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

    result = {}
    for offset in range(BLOCK_SIZE):
        global_idx = BLOCK[0] + offset
        first_in = ffn_inputs.get((global_idx, 1))
        second_in = ffn_inputs.get((global_idx, 2))
        if first_in is not None and second_in is not None:
            result[offset] = (first_in, second_in)
    return result


# =====================================================================
# Compute FFN Jacobian-vector product via finite differences
# J_FFN * delta approx = (FFN(x + eps*delta) - FFN(x)) / eps
# =====================================================================

def compute_jvp_finite_diff(mlp, x, delta, eps=1e-3):
    \"\"\"Compute ||J_FFN * delta|| using finite differences.\"\"\"
    with torch.no_grad():
        out_base = mlp(x)
        out_pert = mlp(x + eps * delta)
        jvp = (out_pert - out_base) / eps
        return jvp.norm().item()


# =====================================================================
# Compute FFN output margin (top-2 gating activation difference)
# =====================================================================

def compute_ffn_margin(mlp, x):
    \"\"\"
    Compute the gating margin gamma: difference between top-2 gate activations.
    Qwen2 uses SiLU-gated MLP: out = down_proj(act(gate_proj(x)) * up_proj(x))
    The 'gate' is gate_proj, and we measure the margin in the activated gate space.
    \"\"\"
    with torch.no_grad():
        gate_out = mlp.gate_proj(x)  # (batch, seq, intermediate)
        gate_activated = torch.nn.functional.silu(gate_out)

        # Per-neuron activation magnitude (average over batch/seq)
        neuron_acts = gate_activated.abs().mean(dim=(0, 1))  # (intermediate,)

        # Top-2 difference
        topk = neuron_acts.topk(2)
        gamma = (topk.values[0] - topk.values[1]).item()
        return gamma, neuron_acts


# =====================================================================
# Compute local key-value conflict C_l
# =====================================================================

def compute_kv_conflict(mlp, gate_activations, top_k=64):
    \"\"\"
    Compute key-value conflict for the top-k most activated neurons.
    Key direction k_r = row of gate_proj (W_gate)
    Value direction v_r = column of down_proj (W_out)
    C_l = mean(key_sim * (1 - value_sim)) for nearest-neighbor pairs.
    \"\"\"
    with torch.no_grad():
        # Get top-k neuron indices
        _, topk_indices = gate_activations.topk(top_k)
        topk_indices = topk_indices.cpu().numpy()

        # Key directions: rows of gate_proj weight
        W_gate = mlp.gate_proj.weight.data  # (intermediate, hidden)
        keys = W_gate[topk_indices]  # (top_k, hidden)
        keys = keys.float()
        keys = keys / keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Value directions: columns of down_proj weight (= rows of down_proj.weight.T)
        # down_proj: (hidden, intermediate), so column r = down_proj.weight[:, r]
        W_down = mlp.down_proj.weight.data  # (hidden, intermediate)
        values = W_down[:, topk_indices].T  # (top_k, hidden)
        values = values.float()
        values = values / values.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Compute pairwise cosine similarities
        key_sim = keys @ keys.T  # (top_k, top_k)
        val_sim = values @ values.T  # (top_k, top_k)

        # For nearest-neighbor pairs: for each neuron, find most similar key neighbor
        # Mask diagonal
        key_sim.fill_diagonal_(0)
        val_sim.fill_diagonal_(0)

        # For each neuron, find the index of most similar key
        nn_indices = key_sim.argmax(dim=-1)  # (top_k,)

        # Gather nearest-neighbor key and value similarities
        nn_key_sims = key_sim[torch.arange(top_k), nn_indices]
        nn_val_sims = val_sim[torch.arange(top_k), nn_indices]

        # C_l = mean(key_sim * (1 - value_sim)) for nearest-neighbor pairs
        conflict = (nn_key_sims * (1 - nn_val_sims)).mean().item()
        return conflict


# =====================================================================
# Compute active-set stability (Jaccard overlap of top-k gate activations
# between first and second pass)
# =====================================================================

def compute_active_set_stability(mlp, ffn_input_first, ffn_input_second, top_k=256):
    \"\"\"Jaccard overlap of top-k activated neurons between first and second pass.\"\"\"
    with torch.no_grad():
        gate_first = mlp.gate_proj(ffn_input_first)
        gate_second = mlp.gate_proj(ffn_input_second)

        # Average over batch and sequence
        act_first = torch.nn.functional.silu(gate_first).abs().mean(dim=(0, 1))
        act_second = torch.nn.functional.silu(gate_second).abs().mean(dim=(0, 1))

        topk_first = set(act_first.topk(top_k).indices.cpu().numpy().tolist())
        topk_second = set(act_second.topk(top_k).indices.cpu().numpy().tolist())

        intersection = len(topk_first & topk_second)
        union = len(topk_first | topk_second)
        jaccard = intersection / max(union, 1)
        return jaccard


# =====================================================================
# Main computation
# =====================================================================

print(f'\nCollecting FFN inputs across {len(CAL_PROMPTS)} calibration prompts...', flush=True)

per_layer_metrics = {offset: {
    'global_layer': BLOCK[0] + offset,
    'jvp_norms': [],
    'margins': [],
    'kv_conflicts': [],
    'jaccard_stabilities': [],
} for offset in range(BLOCK_SIZE)}

for pidx, prompt in enumerate(CAL_PROMPTS):
    print(f'\n  Prompt {pidx}: \"{prompt}\"', flush=True)

    ffn_io = collect_per_layer_ffn_inputs(prompt)

    for offset in range(BLOCK_SIZE):
        if offset not in ffn_io:
            print(f'    WARNING: L{offset} not found in ffn_io', flush=True)
            continue

        ffn_input_first, ffn_input_second = ffn_io[offset]
        delta = ffn_input_second - ffn_input_first
        global_idx = BLOCK[0] + offset
        mlp = original_layers[global_idx].mlp

        # 1. JVP norm
        jvp_norm = compute_jvp_finite_diff(mlp, ffn_input_first, delta)
        per_layer_metrics[offset]['jvp_norms'].append(jvp_norm)

        # 2. FFN margin (on second-pass input, which is where corruption happens)
        gamma, gate_acts = compute_ffn_margin(mlp, ffn_input_second)
        per_layer_metrics[offset]['margins'].append(gamma)

        # 3. Key-value conflict
        kv_conflict = compute_kv_conflict(mlp, gate_acts, top_k=64)
        per_layer_metrics[offset]['kv_conflicts'].append(kv_conflict)

        # 4. Active-set stability
        jaccard = compute_active_set_stability(mlp, ffn_input_first, ffn_input_second, top_k=256)
        per_layer_metrics[offset]['jaccard_stabilities'].append(jaccard)

        print(f'    L{offset} (layer {global_idx}): '
              f'|JVP|={jvp_norm:.2f} gamma={gamma:.4f} C_l={kv_conflict:.4f} '
              f'jaccard={jaccard:.4f}', flush=True)


# =====================================================================
# Compute danger scores and correlate with known harm
# =====================================================================

print(f'\n{\"=\" * 70}')
print('FFN DANGER SCORES')
print(f'{\"=\" * 70}', flush=True)

eps = 1e-6
danger_scores = {}
summary_table = []

print(f'  {\"Layer\":>6s} {\"Global\":>7s} {\"<|JVP|>\":>10s} {\"<gamma>\":>10s} {\"<C_l>\":>10s} '
      f'{\"<Jaccard>\":>10s} {\"R^FFN\":>10s} {\"Known\":>8s} {\"Harm\":>8s}', flush=True)
print(f'  ' + '-' * 90, flush=True)

for offset in range(BLOCK_SIZE):
    m = per_layer_metrics[offset]
    mean_jvp = np.mean(m['jvp_norms'])
    mean_gamma = np.mean(m['margins'])
    mean_conflict = np.mean(m['kv_conflicts'])
    mean_jaccard = np.mean(m['jaccard_stabilities'])

    # R^FFN = C_l * |J_FFN * delta| / (gamma + eps)
    danger = mean_conflict * mean_jvp / (mean_gamma + eps)
    danger_scores[offset] = danger

    known_harm = KNOWN_FFN_HARM[offset]
    harm_level = 'SAFE' if known_harm > -0.5 else ('mild' if known_harm > -2 else 'HIGH')

    print(f'  {offset:>6d} {m[\"global_layer\"]:>7d} {mean_jvp:>10.2f} {mean_gamma:>10.4f} '
          f'{mean_conflict:>10.4f} {mean_jaccard:>10.4f} {danger:>10.4f} {known_harm:>+8.2f} '
          f'{harm_level:>8s}', flush=True)

    summary_table.append({
        'layer_offset': offset,
        'global_layer': m['global_layer'],
        'mean_jvp_norm': float(mean_jvp),
        'mean_margin': float(mean_gamma),
        'mean_kv_conflict': float(mean_conflict),
        'mean_jaccard_stability': float(mean_jaccard),
        'danger_score': float(danger),
        'known_ffn_harm': known_harm,
        'all_jvp_norms': [float(x) for x in m['jvp_norms']],
        'all_margins': [float(x) for x in m['margins']],
        'all_kv_conflicts': [float(x) for x in m['kv_conflicts']],
        'all_jaccard_stabilities': [float(x) for x in m['jaccard_stabilities']],
    })

# =====================================================================
# Correlation analysis
# =====================================================================

print(f'\n{\"=\" * 70}')
print('CORRELATION ANALYSIS')
print(f'{\"=\" * 70}', flush=True)

# Extract arrays for correlation
ds_arr = np.array([danger_scores[i] for i in range(BLOCK_SIZE)])
harm_arr = np.array([KNOWN_FFN_HARM[i] for i in range(BLOCK_SIZE)])
jvp_arr = np.array([np.mean(per_layer_metrics[i]['jvp_norms']) for i in range(BLOCK_SIZE)])
margin_arr = np.array([np.mean(per_layer_metrics[i]['margins']) for i in range(BLOCK_SIZE)])
conflict_arr = np.array([np.mean(per_layer_metrics[i]['kv_conflicts']) for i in range(BLOCK_SIZE)])
jaccard_arr = np.array([np.mean(per_layer_metrics[i]['jaccard_stabilities']) for i in range(BLOCK_SIZE)])

# Note: more negative harm = more harmful FFN. Higher danger score should
# correlate with more negative harm. So we expect negative correlation
# between danger_score and harm (or positive correlation with -harm).

def pearson_corr(x, y):
    if len(x) < 3:
        return 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    if denom < 1e-12:
        return 0.0
    return float((xm * ym).sum() / denom)

def spearman_corr(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return 0.0, 1.0
    r, p = spearmanr(x, y)
    return float(r), float(p)

# We want: high danger -> high harm (more negative known_harm)
# So correlate danger with (-harm) which should be positive
neg_harm = -harm_arr

print(f'  Pearson correlations with actual FFN harmfulness (-harm):')
print(f'    R^FFN (danger score):   r = {pearson_corr(ds_arr, neg_harm):.4f}')
print(f'    |JVP| alone:            r = {pearson_corr(jvp_arr, neg_harm):.4f}')
print(f'    Margin (gamma) alone:   r = {pearson_corr(margin_arr, neg_harm):.4f}')
print(f'    KV conflict (C_l):      r = {pearson_corr(conflict_arr, neg_harm):.4f}')
print(f'    Jaccard instability:    r = {pearson_corr(1 - jaccard_arr, neg_harm):.4f}')

try:
    sp_danger, sp_danger_p = spearman_corr(ds_arr, neg_harm)
    sp_jvp, sp_jvp_p = spearman_corr(jvp_arr, neg_harm)
    sp_jaccard, sp_jaccard_p = spearman_corr(1 - jaccard_arr, neg_harm)

    print(f'\n  Spearman rank correlations with -harm:')
    print(f'    R^FFN:                  rho = {sp_danger:.4f} (p={sp_danger_p:.4f})')
    print(f'    |JVP| alone:            rho = {sp_jvp:.4f} (p={sp_jvp_p:.4f})')
    print(f'    Jaccard instability:    rho = {sp_jaccard:.4f} (p={sp_jaccard_p:.4f})')
except ImportError:
    print(f'  (scipy not available for Spearman correlation)', flush=True)

# =====================================================================
# Identify layers to suppress
# =====================================================================

print(f'\n{\"=\" * 70}')
print('RECOMMENDED FFN SUPPRESSION')
print(f'{\"=\" * 70}', flush=True)

# Sort by danger score descending
sorted_layers = sorted(range(BLOCK_SIZE), key=lambda i: danger_scores[i], reverse=True)
print(f'  Layers ranked by danger score (highest first):', flush=True)
for rank, offset in enumerate(sorted_layers, 1):
    known = KNOWN_FFN_HARM[offset]
    ds = danger_scores[offset]
    agrees = (ds > np.median(ds_arr) and known < -1.5) or (ds <= np.median(ds_arr) and known >= -1.5)
    marker = ' (AGREES)' if agrees else ' (DISAGREES)'
    print(f'    #{rank}: L{offset} (layer {BLOCK[0]+offset}) R^FFN={ds:.4f} '
          f'known_harm={known:+.2f}{marker}', flush=True)

# Suggest suppression threshold
median_danger = np.median(ds_arr)
suppress = [offset for offset in range(BLOCK_SIZE) if danger_scores[offset] > median_danger]
print(f'\n  Suggested suppression (R^FFN > median={median_danger:.4f}):', flush=True)
print(f'    Suppress FFN on layers: {[BLOCK[0]+o for o in suppress]}', flush=True)
print(f'    Keep FFN on layers:     {[BLOCK[0]+o for o in range(BLOCK_SIZE) if o not in suppress]}', flush=True)

# =====================================================================
# Save
# =====================================================================

os.makedirs('results/data/72b/mechanistic', exist_ok=True)
outpath = 'results/data/72b/mechanistic/ffn_danger_scores.json'

with open(outpath, 'w') as f:
    json.dump({
        'date': datetime.now().isoformat(),
        'model': 'calme-2.1-qwen2-72b',
        'block': list(BLOCK),
        'block_size': BLOCK_SIZE,
        'calibration_prompts': CAL_PROMPTS,
        'known_ffn_harm': {str(k): v for k, v in KNOWN_FFN_HARM.items()},
        'per_layer_summary': summary_table,
        'correlations': {
            'pearson_danger_vs_neg_harm': pearson_corr(ds_arr, neg_harm),
            'pearson_jvp_vs_neg_harm': pearson_corr(jvp_arr, neg_harm),
            'pearson_margin_vs_neg_harm': pearson_corr(margin_arr, neg_harm),
            'pearson_conflict_vs_neg_harm': pearson_corr(conflict_arr, neg_harm),
            'pearson_jaccard_instability_vs_neg_harm': pearson_corr(1 - jaccard_arr, neg_harm),
        },
        'recommended_suppress': [BLOCK[0] + o for o in suppress],
    }, f, indent=2)
print(f'\nSaved to {outpath}', flush=True)
print('DONE', flush=True)
"

echo "=== Done at $(date) ==="
