#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_phase1_dla_gem_%j.log
#SBATCH --job-name=dp_dla

# Phase 1: Shared Calibration + DLA Screening + GEM Eigenmask + CIB Scoring
# Runs 48 prompts x 4 modes, collects per-neuron DLA, builds masks

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Phase 1: DLA + GEM + CIB ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from scipy import stats

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache

MODEL_PATH = 'models/full/gemma-3-27b-it'
BLOCKS = [(0, 2), (12, 13), (47, 48)]
SAVE_DIR = 'results/data/gemma3_27b/neuron_analysis'
os.makedirs(SAVE_DIR, exist_ok=True)

# Prompts: mix of reasoning and factual
REASONING_PROMPTS = [
    ('What is 78313086360375 multiplied by 88537453126609?', '6933174468959498727528375'),
    ('What is the cube root of 74088893247?', '4201'),
    ('What is 9999999 multiplied by 9999999?', '99999980000001'),
    ('What is 123456789 multiplied by 987654321?', '121932631112635269'),
    ('What is the square root of 152399025?', '12345'),
    ('What is 7777777 multiplied by 3333333?', '25925923703641'),
    ('What is 456789 raised to the power of 2?', '208655854521'),
    ('What is 11111111 multiplied by 11111111?', '123456787654321'),
    ('If a train travels at 60mph for 2.5 hours, then 80mph for 1.5 hours, total distance?', '270'),
    ('What is 2 raised to the power of 48?', '281474976710656'),
    ('What is 54321 multiplied by 12345?', '670592745'),
    ('What is 314159 multiplied by 271828?', '85397342252'),
]

FACTUAL_PROMPTS = [
    ('What is the capital of Australia?', 'Canberra'),
    ('Who wrote Romeo and Juliet?', 'Shakespeare'),
    ('What is the chemical symbol for gold?', 'Au'),
    ('In what year did World War II end?', '1945'),
    ('What is the largest planet in our solar system?', 'Jupiter'),
    ('Who painted the Mona Lisa?', 'Leonardo'),
    ('What is the speed of light in meters per second?', '299792458'),
    ('What element has atomic number 6?', 'Carbon'),
    ('Who was the first person to walk on the moon?', 'Armstrong'),
    ('What is the boiling point of water in Celsius?', '100'),
    ('What is the capital of Japan?', 'Tokyo'),
    ('How many chromosomes do humans have?', '46'),
]

print(f'Prompts: {len(REASONING_PROMPTS)} reasoning + {len(FACTUAL_PROMPTS)} factual', flush=True)

# Load model
print('Loading model...', flush=True)
model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

# Get unembedding matrix for DLA
lm_head = model.lm_head if hasattr(model, 'lm_head') else None
if lm_head is None:
    # Try nested
    outer = model
    if hasattr(outer, 'language_model'):
        lm_head = outer.language_model.lm_head
    elif hasattr(model.model, 'language_model'):
        lm_head = model.model.language_model.lm_head

print(f'LM head: {lm_head.weight.shape if lm_head else \"NOT FOUND\"}', flush=True)

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

def apply_dup(blocks):
    order = build_order(blocks, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg and hasattr(cfg, 'layer_types') and cfg.layer_types:
            cfg.layer_types = [cfg.layer_types[idx] for idx in order]
        if cfg and hasattr(cfg, 'use_cache'):
            cfg.use_cache = False
    return order

def restore():
    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

# Duplicated layer indices
sorted_blocks = sorted(BLOCKS)
dup_layers = []
for (i, j) in sorted_blocks:
    for l in range(i, j):
        dup_layers.append(l)
print(f'Duplicated layers: {dup_layers}', flush=True)

# ======================================================================
# CALIBRATION: Collect per-neuron activations across 4 modes
# ======================================================================
print('\\n=== Calibration Phase ===', flush=True)

all_prompts = [(p, a, 'reasoning') for p, a in REASONING_PROMPTS] + \
              [(p, a, 'factual') for p, a in FACTUAL_PROMPTS]

def format_prompt(q):
    return f'System: Answer with ONLY the answer, nothing else.\\n\\nUser: {q}\\n\\nAssistant:'

def get_answer_token_id(answer_str):
    toks = tokenizer.encode(answer_str, add_special_tokens=False)
    return toks[0] if toks else None

# For each mode, collect FFN neuron activations on the SECOND pass
modes = {
    'attn_only': {'apply_dup': True, 'ffn_alpha': 0.0},
    'whisper': {'apply_dup': True, 'ffn_alpha': 0.2},
    'full': {'apply_dup': True, 'ffn_alpha': 1.0},
}

# Storage for per-neuron DLA scores
# For each mode, for each prompt, for each dup layer, store FFN neuron activations
neuron_dla = {mode: [] for mode in modes}  # [n_prompts] of {layer: [n_neurons]}

for mode_name, mode_cfg in modes.items():
    print(f'\\nMode: {mode_name}', flush=True)
    order = apply_dup(BLOCKS)

    # Register hooks to capture FFN intermediate activations on second pass
    ffn_data = {}  # layer_idx -> {activations: tensor, fire_count: int}
    hooks = []

    for layer_idx in dup_layers:
        module = original_layers[layer_idx]
        ffn_data[layer_idx] = {'acts': [], 'counter': [0]}

        # Hook on the MLP to capture gate*up product (pre-down-projection)
        counter = ffn_data[layer_idx]['counter']
        acts_list = ffn_data[layer_idx]['acts']
        ffn_alpha = mode_cfg['ffn_alpha']

        def make_ffn_hook(ctr, acts, alpha):
            def hook(module, input, output):
                ctr[0] += 1
                if ctr[0] % 2 == 0:  # second pass
                    # Capture the FFN output before it goes to residual
                    with torch.no_grad():
                        if isinstance(output, tuple):
                            acts.append(output[0][:, -1, :].detach().cpu().float())
                        else:
                            acts.append(output[:, -1, :].detach().cpu().float())
                    # Apply alpha scaling
                    if abs(alpha - 1.0) > 1e-6:
                        if isinstance(output, tuple):
                            return (alpha * output[0],) + output[1:]
                        return alpha * output
                return output
            return hook

        h = module.mlp.register_forward_hook(make_ffn_hook(counter, acts_list, ffn_alpha))
        hooks.append(h)

    # Run prompts
    for idx, (prompt, answer, ptype) in enumerate(all_prompts):
        formatted = format_prompt(prompt)
        inputs = tokenizer(formatted, return_tensors='pt', truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model(inputs['input_ids'], use_cache=False)
            logits = outputs.logits[:, -1, :].float().cpu()

        # Get DLA: project FFN outputs through unembedding to get per-neuron logit contribution
        answer_tok = get_answer_token_id(answer)
        if answer_tok is not None and lm_head is not None:
            unembed_row = lm_head.weight[answer_tok].detach().cpu().float()  # [d_model]
        else:
            unembed_row = None

        prompt_dla = {}
        for layer_idx in dup_layers:
            acts = ffn_data[layer_idx]['acts']
            if acts and unembed_row is not None:
                ffn_out = acts[-1].squeeze(0)  # [d_model]
                # DLA = dot product of FFN output direction with unembedding of answer token
                dla = (ffn_out * unembed_row).sum().item()
                prompt_dla[layer_idx] = dla
            else:
                prompt_dla[layer_idx] = 0.0

        neuron_dla[mode_name].append({
            'prompt': prompt, 'answer': answer, 'type': ptype,
            'dla': prompt_dla,
            'logit_answer': logits[0, answer_tok].item() if answer_tok else 0.0,
        })

        if idx % 8 == 0:
            print(f'  [{idx+1}/{len(all_prompts)}]', flush=True)

    # Clean up hooks and reset counters
    for h in hooks:
        h.remove()
    restore()

print('\\nCalibration complete.', flush=True)

# ======================================================================
# DLA ANALYSIS: Score each layer's FFN contribution
# ======================================================================
print('\\n=== DLA Analysis ===', flush=True)

# Compare whisper vs attn-only to isolate FFN contribution
for layer_idx in dup_layers:
    reasoning_dlas = []
    factual_dlas = []

    for i, entry in enumerate(neuron_dla['whisper']):
        whisper_dla = entry['dla'].get(layer_idx, 0)
        attn_dla = neuron_dla['attn_only'][i]['dla'].get(layer_idx, 0)
        delta = whisper_dla - attn_dla  # FFN's incremental contribution

        if entry['type'] == 'reasoning':
            reasoning_dlas.append(delta)
        else:
            factual_dlas.append(delta)

    r_mean = np.mean(reasoning_dlas) if reasoning_dlas else 0
    f_mean = np.mean(factual_dlas) if factual_dlas else 0

    print(f'  Layer {layer_idx}: reasoning_DLA={r_mean:+.4f} factual_DLA={f_mean:+.4f} '
          f'utility={r_mean - abs(min(0, f_mean)):+.4f}', flush=True)

# ======================================================================
# GEM: Generalized Eigenmask (layer-level, not neuron-level)
# ======================================================================
print('\\n=== GEM Eigenmask ===', flush=True)

# Build feature matrix: for each prompt, the DLA vector across layers
# Z[i, j] = whisper_DLA[layer_j] - attn_only_DLA[layer_j] for prompt i
n_prompts = len(all_prompts)
n_atoms = len(dup_layers)
Z = np.zeros((n_prompts, n_atoms))

delta_scores = []  # whisper score - attn_only score (using logit of answer)
for i in range(n_prompts):
    for j, layer_idx in enumerate(dup_layers):
        w_dla = neuron_dla['whisper'][i]['dla'].get(layer_idx, 0)
        a_dla = neuron_dla['attn_only'][i]['dla'].get(layer_idx, 0)
        Z[i, j] = w_dla - a_dla

    w_logit = neuron_dla['whisper'][i]['logit_answer']
    a_logit = neuron_dla['attn_only'][i]['logit_answer']
    delta_scores.append(w_logit - a_logit)

delta_scores = np.array(delta_scores)

# Positive = whisper helps, negative = whisper hurts
w_pos = np.maximum(delta_scores, 0)
w_neg = np.maximum(-delta_scores, 0)

C_pos = (Z.T * w_pos) @ Z / (w_pos.sum() + 1e-8)
C_neg = (Z.T * w_neg) @ Z / (w_neg.sum() + 1e-8)
C_neg_reg = C_neg + 0.01 * np.eye(n_atoms)

# Generalized eigendecomposition
try:
    from scipy.linalg import eigh
    eigvals, eigvecs = eigh(C_pos, C_neg_reg)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    print(f'Eigenvalues: {eigvals}', flush=True)
    print(f'Top eigenvector (mask weights): {eigvecs[:, 0]}', flush=True)

    # Leverage scores as mask
    V = eigvecs[:, :2]  # top 2
    P = V @ V.T @ C_neg_reg
    gem_mask = np.clip(np.diag(P), 0, 1)
    print(f'GEM mask: {dict(zip(dup_layers, gem_mask))}', flush=True)
except Exception as e:
    print(f'GEM failed: {e}', flush=True)
    gem_mask = np.ones(n_atoms)

# ======================================================================
# CIB: Conditional Information Scoring
# ======================================================================
print('\\n=== CIB Scoring ===', flush=True)

# Simple version: for each atom, compute correlation with help/harm labels
G = (delta_scores > 0).astype(float)  # help indicator
B = (delta_scores < 0).astype(float)  # harm indicator

cib_scores = {}
for j, layer_idx in enumerate(dup_layers):
    z_j = Z[:, j]
    # Point-biserial correlation with help/harm
    if G.sum() > 1 and (1 - G).sum() > 1:
        r_help, _ = stats.pointbiserialr(G, z_j)
    else:
        r_help = 0
    if B.sum() > 1 and (1 - B).sum() > 1:
        r_harm, _ = stats.pointbiserialr(B, z_j)
    else:
        r_harm = 0

    cib = r_help - 1.5 * abs(r_harm)
    cib_scores[layer_idx] = cib
    print(f'  Layer {layer_idx}: r_help={r_help:.3f} r_harm={r_harm:.3f} CIB={cib:.3f}', flush=True)

# ======================================================================
# SAVE RESULTS
# ======================================================================
print('\\n=== Saving ===', flush=True)

results = {
    'n_prompts': n_prompts,
    'dup_layers': dup_layers,
    'neuron_dla': {
        mode: [
            {'type': e['type'], 'dla': {str(k): v for k, v in e['dla'].items()},
             'logit_answer': e['logit_answer']}
            for e in entries
        ]
        for mode, entries in neuron_dla.items()
    },
    'delta_scores': delta_scores.tolist(),
    'gem_eigenmask': dict(zip([str(l) for l in dup_layers], gem_mask.tolist())),
    'gem_eigenvalues': eigvals.tolist() if 'eigvals' in dir() else [],
    'cib_scores': {str(k): v for k, v in cib_scores.items()},
    'per_layer_utility': {},
}

# Per-layer summary
for layer_idx in dup_layers:
    r_dlas = [neuron_dla['whisper'][i]['dla'].get(layer_idx, 0) - neuron_dla['attn_only'][i]['dla'].get(layer_idx, 0)
              for i in range(n_prompts) if all_prompts[i][2] == 'reasoning']
    f_dlas = [neuron_dla['whisper'][i]['dla'].get(layer_idx, 0) - neuron_dla['attn_only'][i]['dla'].get(layer_idx, 0)
              for i in range(n_prompts) if all_prompts[i][2] == 'factual']
    results['per_layer_utility'][str(layer_idx)] = {
        'reasoning_dla_mean': float(np.mean(r_dlas)),
        'factual_dla_mean': float(np.mean(f_dlas)),
        'utility': float(np.mean(r_dlas) - abs(min(0, np.mean(f_dlas)))),
    }

with open(f'{SAVE_DIR}/phase1_dla_gem_cib.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to {SAVE_DIR}/phase1_dla_gem_cib.json', flush=True)
print('\\nPhase 1 complete.', flush=True)
"

echo "=== Finished: $(date) ==="
