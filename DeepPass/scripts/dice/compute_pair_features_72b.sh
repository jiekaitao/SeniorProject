#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_dice_72b_%j.log
#SBATCH --job-name=deeppass_dice72b

cd /blue/cis4914/jietao/DeepPass

/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python -u << 'PYTHON_EOF'
"""
Compute DICE pair features on 72B for top-20 singleton blocks.

For each block:
  - displacement_rho (full-model forward, architecture-agnostic)
  - BLOOD profile (fast: ||layer_output - layer_input|| per layer, 80 layers)
  - Effect matrix (last-token logit delta on calibration prompts, top-1000 tokens)
  - Seam Mahalanobis distance

For each non-overlapping pair:
  - region_dist = |midpoint_a - midpoint_b| / N
  - effect_orth = 1 - CKA(effect_a, effect_b)
  - territory_orth = 1 - cosine(blood_a, blood_b)
  - rho_lift = rho_b_base - rho_b_conditioned_on_a
  - ood_safe = -(seam_md_b_with_a - seam_md_b_base)

Then validate against 23 labeled pairs from 4 JSON files.
"""
import sys, os, json, torch, torch.nn as nn
import numpy as np
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.', 'scripts'))
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

# Top-20 blocks for 72B: extracted from labeled pairs
TOP_BLOCKS = [
    (0, 7), (5, 12), (10, 17), (15, 20), (20, 27),
    (25, 32), (30, 37), (35, 40), (35, 45), (40, 45),
    (40, 47), (45, 50), (45, 52), (50, 55), (50, 60),
    (52, 58), (52, 60), (55, 60), (55, 62), (60, 65),
]

# Calibration prompts -- mix of math and reasoning
CALIBRATION_PROMPTS = [
    "What is 127 * 348?",
    "What is 99999 * 99999?",
    "If a train travels at 60 mph for 2.5 hours, how far does it go?",
    "What is the square root of 1764?",
    "Calculate 15! / 13!",
    "A rectangle has area 48 and perimeter 28. What are its dimensions?",
    "What is 2^16?",
    "If you have 3 red balls and 5 blue balls, what's the probability of drawing 2 red?",
    "What is the sum of all integers from 1 to 100?",
    "A car depreciates 15% per year. After 3 years, what fraction of value remains?",
    "What emotion would someone feel after losing a close friend?",
    "Describe the feeling of watching a sunset after a difficult day.",
    "How would a parent feel seeing their child graduate?",
    "What is the emotional impact of being betrayed by a trusted friend?",
    "Describe the mixed emotions of moving to a new country.",
    "How does nostalgia differ from sadness?",
    "What would someone feel when reunited with a childhood pet?",
    "Describe the emotional complexity of forgiving someone who hurt you.",
    "What is 7^5?",
    "If f(x) = 3x^2 - 2x + 1, what is f(5)?",
    "A store has a 30% sale, then another 20% off. What's the total discount?",
    "What is the derivative of sin(x) * e^x?",
    "Three friends split a $147 bill with 20% tip. How much does each pay?",
    "What is 1/7 as a decimal to 6 places?",
    "How many ways can you arrange the letters in MISSISSIPPI?",
    "A sphere has volume 288pi. What is its radius?",
    "What is the emotional tone of someone saying 'I'm fine' after bad news?",
    "Describe how pride and humility can coexist.",
    "What psychological state leads to procrastination?",
    "How does anticipation differ from anxiety?",
    "What is 13^3?",
    "Calculate the area of a triangle with sides 5, 12, 13.",
]


def build_layer_order(blocks, N):
    """Build execution order for multiple blocks."""
    sorted_blocks = sorted(blocks)
    order = []
    prev = 0
    for (i, j) in sorted_blocks:
        order.extend(list(range(prev, j)))
        order.extend(list(range(i, j)))
        prev = j
    order.extend(list(range(prev, N)))
    return order


def compute_displacement_rho(model, tokenizer, device, block, prompts, N):
    """
    Compute displacement rho for a block using full-model forward passes.
    Architecture-agnostic: uses hooks instead of manual layer iteration.

    rho = ||F(F(h)) - F(h)|| / ||F(h) - h||

    where F is running layers [i,j) and h is the hidden state at layer i.
    """
    inner = model.model
    original_layers = list(inner.layers)
    i, j = block

    rhos = []
    for prompt in prompts[:16]:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)

        with torch.no_grad():
            # Get hidden state at layer i using hooks
            captured = {}

            def capture_input_hook(name):
                def hook_fn(module, input, output):
                    inp = input[0] if isinstance(input, tuple) else input
                    captured[name] = inp.detach().clone()
                return hook_fn

            def capture_output_hook(name):
                def hook_fn(module, input, output):
                    out = output[0] if isinstance(output, tuple) else output
                    captured[name] = out.detach().clone()
                return hook_fn

            # Capture input to layer i (= output of layer i-1, or embed if i=0)
            hooks = []
            if i == 0:
                # Hook on embed_tokens output
                def embed_hook(module, input, output):
                    captured['h_at_i'] = output.detach().clone()
                hooks.append(inner.embed_tokens.register_forward_hook(embed_hook))
            else:
                # Capture output of layer i-1
                hooks.append(inner.layers[i-1].register_forward_hook(capture_output_hook('h_at_i')))

            # Capture output of layer j-1 (= h after first pass through block)
            hooks.append(inner.layers[j-1].register_forward_hook(capture_output_hook('h_at_j')))

            # Run full forward to get hidden states
            model(**ids, use_cache=False)

            for h in hooks:
                h.remove()

            if 'h_at_i' not in captured or 'h_at_j' not in captured:
                continue

            h_input = captured['h_at_i']  # h before block
            h_after_first = captured['h_at_j']  # h after block (first pass)

            # Now we need to run layers [i,j) a second time on h_after_first.
            # Use position embeddings from a clean forward.
            # For Qwen2 architecture, layers need position_embeddings.
            # Get them by running embed + rotary.
            h_for_pos = inner.embed_tokens(ids['input_ids'])
            seq_len = h_for_pos.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h_for_pos, pos_ids)

            # Second pass: F(F(h))
            h2 = h_after_first.clone()
            for layer_idx in range(i, j):
                out = original_layers[layer_idx](h2, position_embeddings=pos_embeds, use_cache=False)
                h2 = out[0] if isinstance(out, tuple) else out

            num = torch.norm(h2 - h_after_first).item()
            den = torch.norm(h_after_first - h_input).item()
            if den > 1e-8:
                rhos.append(num / den)

    return np.mean(rhos) if rhos else 1.0


def compute_blood_profile_fast(model, tokenizer, device, prompts, N):
    """
    Fast BLOOD: ||layer_output - layer_input|| per layer, averaged over prompts.
    Uses hooks for architecture safety. Only 8 prompts for 72B efficiency.
    Returns array of shape [N].
    """
    inner = model.model
    layer_norms = [[] for _ in range(N)]
    hooks = []

    def make_hook(idx):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            norm = torch.norm(out.float() - inp.float()).item()
            layer_norms[idx].append(norm)
        return hook_fn

    for idx, layer in enumerate(inner.layers):
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    for prompt in prompts[:8]:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            model(ids['input_ids'], use_cache=False)

    for h in hooks:
        h.remove()

    return np.array([float(np.mean(ns)) if ns else 0.0 for ns in layer_norms])


def compute_blood_profile_for_block(model, tokenizer, device, prompts, N, block):
    """
    Compute BLOOD profile when a specific block is duplicated.
    Sets up the duplicated layer order, runs forward, measures per-layer norms.
    Returns array of shape [N] (mapped back to original layer indices).
    """
    inner = model.model
    original_layers = list(inner.layers)
    order = build_layer_order([block], N)

    # Temporarily rearrange layers
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)

    # Compute BLOOD on rearranged model
    layer_norms = [[] for _ in range(len(order))]
    hooks = []

    def make_hook(idx):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            norm = torch.norm(out.float() - inp.float()).item()
            layer_norms[idx].append(norm)
        return hook_fn

    for idx in range(len(order)):
        hooks.append(inner.layers[idx].register_forward_hook(make_hook(idx)))

    for prompt in prompts[:8]:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            model(ids['input_ids'], use_cache=False)

    for h in hooks:
        h.remove()

    # Restore
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    # Map back to original layer indices (average duplicated layers)
    profile = np.zeros(N)
    counts = np.zeros(N)
    for step_idx, orig_idx in enumerate(order):
        val = float(np.mean(layer_norms[step_idx])) if layer_norms[step_idx] else 0.0
        profile[orig_idx] += val
        counts[orig_idx] += 1
    counts[counts == 0] = 1
    profile /= counts

    return profile


def compute_effect_matrix(model, tokenizer, device, N, block, prompts):
    """
    Compute last-token logit delta for each calibration prompt when block is duplicated.
    Returns matrix of shape [n_prompts, vocab_subset].
    Uses top-1000 vocab tokens for efficiency.
    Uses 16 prompts for effect matrix.
    """
    inner = model.model
    original_layers = list(inner.layers)
    VOCAB_SLICE = 1000

    # Baseline logits (full model, no duplication)
    baseline_logits = []
    topk_indices = None
    with torch.no_grad():
        for prompt in prompts[:16]:
            ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
            out = model(**ids, use_cache=False)
            logits = out.logits[0, -1, :].float()
            if topk_indices is None:
                topk_indices = logits.abs().topk(VOCAB_SLICE).indices
            baseline_logits.append(logits[topk_indices].cpu().numpy())

    # Duplicated logits
    order = build_layer_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)

    dup_logits = []
    with torch.no_grad():
        for prompt in prompts[:16]:
            ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
            out = model(**ids, use_cache=False)
            logits = out.logits[0, -1, :].float()
            dup_logits.append(logits[topk_indices].cpu().numpy())

    # Restore
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    baseline_mat = np.array(baseline_logits)
    dup_mat = np.array(dup_logits)
    effect = dup_mat - baseline_mat

    return effect


def compute_seam_mahalanobis(model, tokenizer, device, N, block, prompts):
    """
    Compute Mahalanobis distance at the exit seam of a duplicated block.
    Measures how OOD the post-duplication hidden state is vs base model.
    Uses hooks for architecture safety.
    """
    inner = model.model
    original_layers = list(inner.layers)
    i, j = block

    # Collect base model hidden states at layer j (last token)
    base_states = []

    def make_capture_hook(storage):
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            storage.append(out[:, -1, :].float().detach().cpu())
        return hook_fn

    for prompt in prompts[:8]:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        capture = []
        hook = inner.layers[j-1].register_forward_hook(make_capture_hook(capture))
        with torch.no_grad():
            model(**ids, use_cache=False)
        hook.remove()
        if capture:
            base_states.append(capture[0])

    if not base_states:
        return 0.0

    base_stack = torch.cat(base_states, dim=0)
    mu = base_stack.mean(dim=0)
    var = base_stack.var(dim=0).clamp(min=1e-6)

    # Collect duplicated model hidden states at exit seam
    order = build_layer_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)

    # The exit seam: after the duplicated block, we need the hidden state
    # at step position j + (j-i) in the order (end of second pass).
    # The second pass of block [i,j) starts at position j in the order
    # and ends at position j + (j-i) - 1. So we want step j + (j-i) - 1.
    seam_step = j + (j - i) - 1  # Index in order list

    # Since the same physical layer module appears twice in the rearranged list,
    # a hook fires for BOTH occurrences. We need the 2nd firing (exit of 2nd pass).
    # Count how many times layer j-1 appears before and at seam_step.
    target_layer_idx = order[seam_step]  # = j-1
    n_occurrences_before = sum(1 for k in range(seam_step + 1) if order[k] == target_layer_idx)

    dup_states = []
    for prompt in prompts[:8]:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        capture = []
        call_counter = [0]

        def make_seam_hook(storage, target_call, counter):
            def hook_fn(module, input, output):
                counter[0] += 1
                if counter[0] == target_call:
                    out = output[0] if isinstance(output, tuple) else output
                    storage.append(out[:, -1, :].float().detach().cpu())
            return hook_fn

        if seam_step < len(inner.layers):
            hook = inner.layers[seam_step].register_forward_hook(
                make_seam_hook(capture, n_occurrences_before, call_counter))
        with torch.no_grad():
            model(**ids, use_cache=False)
        if seam_step < len(inner.layers):
            hook.remove()
        if capture:
            dup_states.append(capture[0])

    # Restore
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    if not dup_states:
        return 0.0

    dup_stack = torch.cat(dup_states, dim=0)
    md = ((dup_stack - mu) ** 2 / var).sum(dim=-1).sqrt().mean().item()
    return md


def compute_rho_conditioned(model, tokenizer, device, N, block_a, block_b, prompts):
    """
    Compute displacement rho of block_b when block_a is already applied.
    Uses full-model forward with block_a duplicated, then manually runs block_b twice.
    """
    inner = model.model
    original_layers = list(inner.layers)
    i_b, j_b = block_b

    # Set up model with block_a applied
    order_a = build_layer_order([block_a], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order_a])
    model.config.num_hidden_layers = len(order_a)

    rhos = []
    for prompt in prompts[:16]:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)

        with torch.no_grad():
            # We need hidden state at block_b's start in the modified model.
            # Run full modified model up to step min(i_b, len(order_a)),
            # then manually run block_b twice.

            h = inner.embed_tokens(ids['input_ids'])
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            # Run current model up to the step corresponding to layer i_b
            current_N = len(inner.layers)
            for step_idx in range(min(i_b, current_N)):
                out = inner.layers[step_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

            h_input = h.clone()

            # First pass through block_b using original layers
            h1 = h_input.clone()
            for layer_idx in range(i_b, j_b):
                out = original_layers[layer_idx](h1, position_embeddings=pos_embeds, use_cache=False)
                h1 = out[0] if isinstance(out, tuple) else out

            # Second pass
            h2 = h1.clone()
            for layer_idx in range(i_b, j_b):
                out = original_layers[layer_idx](h2, position_embeddings=pos_embeds, use_cache=False)
                h2 = out[0] if isinstance(out, tuple) else out

            num = torch.norm(h2 - h1).item()
            den = torch.norm(h1 - h_input).item()
            if den > 1e-8:
                rhos.append(num / den)

    # Restore
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    return np.mean(rhos) if rhos else 1.0


def compute_seam_md_conditioned(model, tokenizer, device, N, block_a, block_b, prompts):
    """
    Compute seam Mahalanobis for block_b when block_a is already applied.
    Base stats come from original model; measurement from model+block_a+block_b.
    """
    inner = model.model
    original_layers = list(inner.layers)
    i_b, j_b = block_b

    # Base stats: original model hidden states at layer j_b
    base_states = []

    def make_capture_hook(storage):
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            storage.append(out[:, -1, :].float().detach().cpu())
        return hook_fn

    for prompt in prompts[:8]:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        capture = []
        hook = inner.layers[j_b-1].register_forward_hook(make_capture_hook(capture))
        with torch.no_grad():
            model(**ids, use_cache=False)
        hook.remove()
        if capture:
            base_states.append(capture[0])

    if not base_states:
        return 0.0

    base_stack = torch.cat(base_states, dim=0)
    mu = base_stack.mean(dim=0)
    var = base_stack.var(dim=0).clamp(min=1e-6)

    # Now set up model with block_a, then run block_b duplicated on top
    order_a = build_layer_order([block_a], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order_a])
    model.config.num_hidden_layers = len(order_a)

    dup_states = []
    for prompt in prompts[:8]:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)

        with torch.no_grad():
            h = inner.embed_tokens(ids['input_ids'])
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            # Run current (block_a applied) model up to j_b
            current_N = len(inner.layers)
            for step_idx in range(min(j_b, current_N)):
                out = inner.layers[step_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

            # Run block_b a second time (layers i_b..j_b using original layers)
            for layer_idx in range(i_b, j_b):
                out = original_layers[layer_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

            dup_states.append(h[:, -1, :].float().cpu())

    # Restore
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    if not dup_states:
        return 0.0

    dup_stack = torch.cat(dup_states, dim=0)
    md = ((dup_stack - mu) ** 2 / var).sum(dim=-1).sqrt().mean().item()
    return md


def linear_cka(X, Y):
    """
    Linear CKA between two matrices X [n, p] and Y [n, q].
    Kornblith et al. 2019.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y

    hsic_xy = np.sum(XtY ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(dot / (na * nb))


def load_labeled_pairs():
    """
    Load all 23 labeled pairs from 4 JSON files.
    Returns list of dicts with keys: block_a, block_b, combined_score.
    """
    labeled = []
    baseline_combined = None

    # 1. 72b_best_pairs_dual_probe.json -- list with name, math, eq, combined
    with open('results/data/72b/pairs/72b_best_pairs_dual_probe.json') as f:
        data = json.load(f)
    for entry in data:
        if entry['name'] == 'baseline':
            baseline_combined = entry['combined']
            continue
        # Parse name like "(15,20)+(50,60)"
        parts = entry['name'].split('+')
        if len(parts) != 2:
            continue
        a = tuple(int(x) for x in parts[0].strip('()').split(','))
        b = tuple(int(x) for x in parts[1].strip('()').split(','))
        labeled.append({
            'block_a': list(a), 'block_b': list(b),
            'combined': entry['combined'],
            'math': entry.get('math', None),
            'source': 'dual_probe',
        })

    # 2. 72b_cross_region_pairs.json -- list with a, b, score
    with open('results/data/72b/pairs/72b_cross_region_pairs.json') as f:
        data = json.load(f)
    for entry in data:
        labeled.append({
            'block_a': entry['a'], 'block_b': entry['b'],
            'combined': entry['score'],  # math score used as combined proxy
            'math': entry['score'],
            'source': 'cross_region',
        })

    # 3. 72b_pair_sweep.json -- dict with baseline, pairs list
    with open('results/data/72b/pairs/72b_pair_sweep.json') as f:
        data = json.load(f)
    if baseline_combined is None:
        baseline_combined = data.get('baseline', 0.6301)
    for entry in data['pairs']:
        labeled.append({
            'block_a': entry['a'], 'block_b': entry['b'],
            'combined': entry['score'],  # math score
            'math': entry['score'],
            'delta': entry.get('delta', None),
            'source': 'pair_sweep',
        })

    # 4. more_pairs_round2.json -- list with blocks, name, math, eq, combined
    with open('results/data/72b/pairs/more_pairs_round2.json') as f:
        data = json.load(f)
    for entry in data:
        blocks = entry['blocks']
        labeled.append({
            'block_a': blocks[0], 'block_b': blocks[1],
            'combined': entry['combined'],
            'math': entry.get('math', None),
            'source': 'round2',
        })

    # Deduplicate by (block_a, block_b) -- keep the one with 'combined' key if available
    seen = {}
    for p in labeled:
        key = (tuple(p['block_a']), tuple(p['block_b']))
        rev_key = (tuple(p['block_b']), tuple(p['block_a']))
        if key not in seen and rev_key not in seen:
            seen[key] = p
        else:
            # Keep the one with higher-quality score (prefer dual_probe/round2 over math-only)
            existing_key = key if key in seen else rev_key
            existing = seen[existing_key]
            if p['source'] in ('dual_probe', 'round2') and existing['source'] not in ('dual_probe', 'round2'):
                seen[existing_key] = p

    deduped = list(seen.values())
    print(f"  Loaded {len(deduped)} unique labeled pairs (from {len(labeled)} total entries)")
    return deduped, baseline_combined


def main():
    print("=" * 60)
    print("DICE Feature Computation (72B)")
    print("=" * 60)

    model, tokenizer = load_original_model(MODEL_PATH)
    device = next(model.parameters()).device
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)
    print(f"Loaded model: {MODEL_PATH} ({N} layers)")
    assert N == 80, f"Expected 80 layers for 72B, got {N}"

    prompts = CALIBRATION_PROMPTS

    # ============================================
    # Phase 0: Baseline BLOOD profile
    # ============================================
    print("\n--- Phase 0: Baseline BLOOD profile (no duplication) ---")
    baseline_blood = compute_blood_profile_fast(model, tokenizer, device, prompts, N)
    print(f"  Baseline BLOOD computed. Max norm at layer {np.argmax(baseline_blood)} = {np.max(baseline_blood):.1f}")
    torch.cuda.empty_cache()

    # ============================================
    # Phase 1: Per-block features
    # ============================================
    print("\n--- Phase 1: Per-block features ---")

    block_data = {}
    for block_idx, block in enumerate(TOP_BLOCKS):
        print(f"\n  [{block_idx+1}/{len(TOP_BLOCKS)}] Block {block}:")

        # Displacement rho (on base model, full-forward)
        rho = compute_displacement_rho(model, tokenizer, device, block, prompts, N)
        print(f"    displacement_rho = {rho:.4f}")
        torch.cuda.empty_cache()

        # BLOOD profile (for this block's duplication)
        blood_profile = compute_blood_profile_for_block(model, tokenizer, device, prompts, N, block)
        print(f"    BLOOD profile computed (non-zero layers: {(blood_profile > 0).sum()}/{N})")
        torch.cuda.empty_cache()

        # Effect matrix (logit deltas, 16 prompts)
        effect_mat = compute_effect_matrix(model, tokenizer, device, N, block, prompts)
        print(f"    Effect matrix: {effect_mat.shape}")
        torch.cuda.empty_cache()

        # Seam Mahalanobis
        seam_md = compute_seam_mahalanobis(model, tokenizer, device, N, block, prompts)
        print(f"    Seam Mahalanobis = {seam_md:.2f}")
        torch.cuda.empty_cache()

        block_data[block] = {
            'rho': rho,
            'blood_profile': blood_profile,
            'effect_matrix': effect_mat,
            'seam_md': seam_md,
            'midpoint': (block[0] + block[1]) / 2,
        }

    # ============================================
    # Phase 2: Pairwise interaction features
    # ============================================
    print("\n--- Phase 2: Pairwise interaction features ---")

    pair_features = []

    sorted_blocks = sorted(TOP_BLOCKS, key=lambda b: b[0])
    valid_pairs = [(a, b) for a, b in combinations(sorted_blocks, 2)
                   if a[1] <= b[0]]  # Non-overlapping

    print(f"  Computing features for {len(valid_pairs)} non-overlapping pairs...")

    for pair_idx, (a, b) in enumerate(valid_pairs):
        da = block_data[a]
        db = block_data[b]

        # 1. Region distance
        region_dist = abs(da['midpoint'] - db['midpoint']) / N

        # 2. Effect orthogonality (1 - CKA)
        cka = linear_cka(da['effect_matrix'], db['effect_matrix'])
        effect_orth = 1.0 - cka

        # 3. BLOOD-territory orthogonality (1 - cosine)
        blood_cos = cosine_sim(da['blood_profile'], db['blood_profile'])
        territory_orth = 1.0 - blood_cos

        # 4. Conditional rho lift: apply block a, then measure rho of b
        rho_b_given_a = compute_rho_conditioned(model, tokenizer, device, N, a, b, prompts)
        rho_lift = db['rho'] - rho_b_given_a  # Positive = b is more contractive after a
        torch.cuda.empty_cache()

        # 5. Conditional seam Mahalanobis change
        md_b_given_a = compute_seam_md_conditioned(model, tokenizer, device, N, a, b, prompts)
        ood_safe = -(md_b_given_a - db['seam_md'])  # Negative change = safer
        torch.cuda.empty_cache()

        features = {
            'block_a': list(a),
            'block_b': list(b),
            'region_dist': region_dist,
            'effect_orth': effect_orth,
            'territory_orth': territory_orth,
            'rho_lift': rho_lift,
            'ood_safe': ood_safe,
            'cka': cka,
            'blood_cos': blood_cos,
            'rho_b_base': db['rho'],
            'rho_b_cond': rho_b_given_a,
            'md_b_base': db['seam_md'],
            'md_b_cond': md_b_given_a,
        }
        pair_features.append(features)

        print(f"  [{pair_idx+1}/{len(valid_pairs)}] "
              f"{str(a):12s} -> {str(b):12s}: "
              f"dist={region_dist:.2f} eff_orth={effect_orth:.3f} "
              f"terr_orth={territory_orth:.3f} rho_lift={rho_lift:+.4f} "
              f"ood_safe={ood_safe:+.1f}")

    # ============================================
    # Phase 3: Validate against labeled pairs
    # ============================================
    print("\n--- Phase 3: Validation against labeled pairs ---")

    labeled_pairs, baseline_combined = load_labeled_pairs()
    print(f"  Baseline combined score: {baseline_combined}")

    # Match computed features to labeled pairs
    feature_lookup = {}
    for pf in pair_features:
        key = (tuple(pf['block_a']), tuple(pf['block_b']))
        feature_lookup[key] = pf

    matched = []
    for lp in labeled_pairs:
        a = tuple(lp['block_a'])
        b = tuple(lp['block_b'])
        key = (a, b)
        rev_key = (b, a)
        if key in feature_lookup:
            matched.append((feature_lookup[key], lp))
        elif rev_key in feature_lookup:
            matched.append((feature_lookup[rev_key], lp))

    print(f"  Matched {len(matched)} / {len(labeled_pairs)} labeled pairs with computed features")

    validation_results = None

    if len(matched) < 3:
        print("  WARNING: Too few matched pairs for meaningful validation.")
    else:
        # Rank-normalize features
        def _rank01(x):
            x = np.asarray(x, dtype=np.float64)
            n = len(x)
            if n <= 1:
                return np.array([0.5] * n)
            order = x.argsort().argsort()
            return order / (n - 1)

        feature_names = ['region_dist', 'effect_orth', 'territory_orth', 'rho_lift', 'ood_safe']
        feature_arrays = {name: np.array([m[0][name] for m in matched]) for name in feature_names}
        ranked = {name: _rank01(arr) for name, arr in feature_arrays.items()}

        # Default theory-signed weights (no fitting)
        default_weights = {
            "rho_lift": 1.00,
            "effect_orth": 1.00,
            "territory_orth": 0.75,
            "region_dist": 0.50,
            "ood_safe": 0.75,
        }

        # Compute edge scores with default weights
        edge_scores_pred = []
        combined_obs = []
        for idx, (pf, gt_info) in enumerate(matched):
            features_ranked = {name: ranked[name][idx] for name in feature_names}
            es = sum(default_weights.get(k, 0.0) * features_ranked.get(k, 0.0) for k in feature_names)
            edge_scores_pred.append(es)
            combined_obs.append(gt_info['combined'])

        edge_scores_pred = np.array(edge_scores_pred)
        combined_obs = np.array(combined_obs)

        # Spearman correlation with combined scores
        from scipy.stats import spearmanr
        r, p = spearmanr(edge_scores_pred, combined_obs)
        print(f"\n  Spearman(predicted_edge, observed_combined) = {r:.3f} (p={p:.4f})")

        # AUROC for predicting "good pair" (above median combined score)
        median_combined = np.median(combined_obs)
        good_pair = (combined_obs > median_combined).astype(int)
        print(f"  Median combined score: {median_combined:.4f}")
        print(f"  Good pairs (above median): {good_pair.sum()} / {len(good_pair)}")

        if good_pair.sum() > 0 and good_pair.sum() < len(good_pair):
            try:
                from sklearn.metrics import roc_auc_score
                auroc = roc_auc_score(good_pair, edge_scores_pred)
                print(f"  AUROC(good pair) = {auroc:.3f}")
            except ImportError:
                # Manual AUROC
                pos = edge_scores_pred[good_pair == 1]
                neg = edge_scores_pred[good_pair == 0]
                auroc = np.mean([float(p_val > n_val) + 0.5 * float(p_val == n_val)
                                 for p_val in pos for n_val in neg])
                print(f"  AUROC(good pair) = {auroc:.3f} (manual)")
        else:
            auroc = float('nan')
            print(f"  AUROC: cannot compute (all pairs same class)")

        # Top-k precision
        k = min(5, len(matched))
        top_k_pred = np.argsort(-edge_scores_pred)[:k]
        top_k_good = sum(good_pair[i] for i in top_k_pred)
        print(f"  Top-{k} precision (good pairs in top-{k} predicted): {top_k_good}/{k}")

        # Individual feature correlations
        print("\n  Individual feature correlations with combined score:")
        for name in feature_names:
            r_feat, p_feat = spearmanr(feature_arrays[name], combined_obs)
            print(f"    {name:20s}: r={r_feat:+.3f} (p={p_feat:.4f})")

        validation_results = {
            'spearman_r': float(r),
            'spearman_p': float(p),
            'auroc': float(auroc) if not np.isnan(auroc) else None,
            'n_matched': len(matched),
            'n_labeled': len(labeled_pairs),
            'median_combined': float(median_combined),
            'baseline_combined': float(baseline_combined) if baseline_combined else None,
        }

    # ============================================
    # Save everything
    # ============================================
    print("\n--- Saving results ---")

    save_data = {
        'model': MODEL_PATH,
        'n_layers': N,
        'n_blocks': len(TOP_BLOCKS),
        'top_blocks': [list(b) for b in TOP_BLOCKS],
        'blocks': {str(k): {
            'rho': v['rho'],
            'seam_md': v['seam_md'],
            'midpoint': v['midpoint'],
            'blood_profile': v['blood_profile'].tolist(),
        } for k, v in block_data.items()},
        'baseline_blood_profile': baseline_blood.tolist(),
        'pair_features': pair_features,
        'labeled_pairs': labeled_pairs,
        'validation': validation_results if validation_results is not None else {
            'n_matched': len(matched),
            'n_labeled': len(labeled_pairs),
            'error': 'too few matched pairs',
        },
    }

    os.makedirs('results/data/72b/dice', exist_ok=True)
    with open('results/data/72b/dice/72b_pair_features.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"  Saved to results/data/72b/dice/72b_pair_features.json")
    print(f"  Blocks computed: {len(block_data)}")
    print(f"  Pair features computed: {len(pair_features)}")
    print(f"  Labeled pairs matched: {len(matched)}")
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
PYTHON_EOF
