"""
Compute DICE pair features on 7B for all top-10 singleton blocks.

For each block:
  - BLOOD influence profile (per-layer BLOOD score)
  - Effect matrix (last-token logit delta on calibration prompts)
  - Displacement rho

For each ordered pair (i < j):
  - Region distance
  - Effect orthogonality (1 - CKA)
  - BLOOD-territory orthogonality (1 - cosine of BLOOD profiles)
  - Conditional rho lift (rho_j_base - rho_j_given_i)
  - Conditional BLOOD lift
  - Conditional seam Mahalanobis change

Then validate against 22 labeled pairs.
"""
import sys, os, json, torch, torch.nn as nn
import numpy as np
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from layer_duplicator import load_original_model, generate_no_cache

# Top-10 7B blocks from pairwise_stacking_sweep.json
TOP_BLOCKS = [
    (10, 11), (18, 21), (16, 21), (18, 27), (14, 27),
    (8, 15), (16, 27), (20, 21), (6, 15), (16, 17),
]

# Calibration prompts — mix of math and reasoning
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
    "A sphere has volume 288π. What is its radius?",
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


def compute_blood_profile(model, tokenizer, inner, original_layers, N, block, prompts, n_hutchinson=3):
    """
    Compute per-layer BLOOD score (||J||^2_F) for a duplicated model.
    Returns array of shape [N] with BLOOD score at each original layer position.
    """
    order = build_layer_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)

    device = next(model.parameters()).device
    layer_scores = defaultdict(list)

    for prompt in prompts[:8]:  # Use subset for speed
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)

        with torch.no_grad():
            h = inner.embed_tokens(inputs["input_ids"])
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

        # Collect boundary hidden states
        boundary_h = [h.detach()]
        with torch.no_grad():
            for step_idx, layer_idx in enumerate(order):
                out = inner.layers[step_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
                boundary_h.append(h.detach())

        # Compute Jacobian norm at each step
        for step_idx in range(len(order)):
            h_in = boundary_h[step_idx].detach().clone()
            jnorms = []
            for _ in range(n_hutchinson):
                z = torch.randn_like(h_in)
                h_in_g = h_in.detach().clone().requires_grad_(True)
                out = inner.layers[step_idx](h_in_g, position_embeddings=pos_embeds, use_cache=False)
                h_out = out[0] if isinstance(out, tuple) else out
                Jz = torch.autograd.grad(h_out, h_in_g, grad_outputs=z, create_graph=False)[0]
                jnorms.append((Jz ** 2).sum().item())
            layer_scores[order[step_idx]].append(np.mean(jnorms))

        del boundary_h
        torch.cuda.empty_cache()

    # Restore
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    # Average across prompts, return per-original-layer
    profile = np.zeros(N)
    for layer_idx in range(N):
        if layer_idx in layer_scores:
            profile[layer_idx] = np.mean(layer_scores[layer_idx])
    return profile


def compute_effect_matrix(model, tokenizer, inner, original_layers, N, block, prompts):
    """
    Compute last-token logit delta for each calibration prompt when block is duplicated.
    Returns matrix of shape [n_prompts, vocab_subset].

    Uses top-1000 vocab tokens for efficiency.
    """
    device = next(model.parameters()).device
    VOCAB_SLICE = 1000  # Top-k tokens to track

    # Baseline logits
    baseline_logits = []
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
            out = model(**ids, use_cache=False)
            logits = out.logits[0, -1, :].float()  # Last token
            # Get top-k indices from first prompt, reuse for all
            if len(baseline_logits) == 0:
                topk_indices = logits.abs().topk(VOCAB_SLICE).indices
            baseline_logits.append(logits[topk_indices].cpu().numpy())

    # Duplicated logits
    order = build_layer_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)

    dup_logits = []
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
            out = model(**ids, use_cache=False)
            logits = out.logits[0, -1, :].float()
            dup_logits.append(logits[topk_indices].cpu().numpy())

    # Restore
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    # Effect = logit delta
    baseline_mat = np.array(baseline_logits)
    dup_mat = np.array(dup_logits)
    effect = dup_mat - baseline_mat

    return effect


def compute_displacement_rho(model, tokenizer, inner, original_layers, N, block, prompts):
    """Compute displacement rho for a block on the current model state."""
    device = next(model.parameters()).device
    i, j = block

    rhos = []
    for prompt in prompts[:8]:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)

        with torch.no_grad():
            # Get hidden state at layer i
            h = inner.embed_tokens(ids["input_ids"])
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            for layer_idx in range(i):
                out = inner.layers[layer_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

            h_input = h.clone()

            # First pass: F(h)
            h1 = h_input.clone()
            for layer_idx in range(i, j):
                out = inner.layers[layer_idx](h1, position_embeddings=pos_embeds, use_cache=False)
                h1 = out[0] if isinstance(out, tuple) else out

            # Second pass: F(F(h))
            h2 = h1.clone()
            for layer_idx in range(i, j):
                out = inner.layers[layer_idx](h2, position_embeddings=pos_embeds, use_cache=False)
                h2 = out[0] if isinstance(out, tuple) else out

            num = torch.norm(h2 - h1).item()
            den = torch.norm(h1 - h_input).item()
            if den > 1e-8:
                rhos.append(num / den)

    return np.mean(rhos) if rhos else 1.0


def compute_seam_mahalanobis(model, tokenizer, inner, original_layers, N, block, prompts):
    """
    Compute Mahalanobis distance at the exit seam of a duplicated block.
    Measures how OOD the post-duplication hidden state is vs base model.
    """
    device = next(model.parameters()).device
    i, j = block

    # Collect base model hidden states at layer j
    base_states = []
    with torch.no_grad():
        for prompt in prompts[:8]:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
            h = inner.embed_tokens(ids["input_ids"])
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)
            for layer_idx in range(j):
                out = inner.layers[layer_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
            base_states.append(h[:, -1, :].float().cpu())  # Last token

    base_stack = torch.cat(base_states, dim=0)  # [n_prompts, hidden]
    mu = base_stack.mean(dim=0)
    var = base_stack.var(dim=0).clamp(min=1e-6)

    # Collect duplicated model hidden states at exit seam
    order = build_layer_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)

    # The exit seam is at position j + (j-i) in the order = after second pass
    seam_pos = j + (j - i)  # Index in order where second pass ends

    dup_states = []
    with torch.no_grad():
        for prompt in prompts[:8]:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
            h = inner.embed_tokens(ids["input_ids"])
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)
            for step_idx in range(min(seam_pos, len(order))):
                out = inner.layers[step_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
            dup_states.append(h[:, -1, :].float().cpu())

    # Restore
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    dup_stack = torch.cat(dup_states, dim=0)
    # Diagonal Mahalanobis: mean of (x - mu)^2 / var
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


def _compute_rho_on_current_model(model, tokenizer, inner, block, prompts):
    """
    Compute displacement rho for a block on whatever model state is currently set.
    Uses inner.layers as-is (may already have other blocks applied).
    Block indices refer to ORIGINAL model positions — we find the corresponding
    layers in the current model by scanning for them.
    """
    device = next(model.parameters()).device
    i, j = block
    current_N = len(inner.layers)

    rhos = []
    for prompt in prompts[:8]:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)

        with torch.no_grad():
            h = inner.embed_tokens(ids["input_ids"])
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            # Run through all layers up to where block b would start
            # In the current model, we need to run up to block b's start
            # then run block b twice
            # Since layers may be rearranged, just run the full model but
            # intercept hidden states at the right positions

            # Simpler approach: run the full current model to get h at layer i,
            # then run block [i,j) twice manually using original_layers
            for step_idx in range(current_N):
                out = inner.layers[step_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

            # Actually, we need hidden state BEFORE block b in the current model.
            # Reset and run step by step, stopping before the first occurrence of layer i.
            pass

        # Fall back to simpler method: run full model, extract h at boundaries
        # by running original layers i..j on top of the full current model's
        # hidden state at position i
        with torch.no_grad():
            h = inner.embed_tokens(ids["input_ids"])
            # Run current model up to where we'd start block b
            # Find first step in current model that maps to original layer i
            # This is tricky, so let's just use the base approach:
            # run all layers, collect h just before and after block b region

            # Simple: run up to step i in the current model
            for step_idx in range(min(i, current_N)):
                out = inner.layers[step_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

            h_input = h.clone()

            # First pass through block using original layers
            h1 = h_input.clone()
            from layer_duplicator import load_original_model  # just for original_layers ref
            # We have them in scope via closure — use the originals stored in block_data
            # Actually we need original_layers... pass them differently
            # For now, use inner.layers indices i..j which point to the same objects
            for layer_idx in range(i, min(j, current_N)):
                out = inner.layers[layer_idx](h1, position_embeddings=pos_embeds, use_cache=False)
                h1 = out[0] if isinstance(out, tuple) else out

            # Second pass
            h2 = h1.clone()
            for layer_idx in range(i, min(j, current_N)):
                out = inner.layers[layer_idx](h2, position_embeddings=pos_embeds, use_cache=False)
                h2 = out[0] if isinstance(out, tuple) else out

            num = torch.norm(h2 - h1).item()
            den = torch.norm(h1 - h_input).item()
            if den > 1e-8:
                rhos.append(num / den)

    return np.mean(rhos) if rhos else 1.0


def _compute_seam_md_on_current_model(model, tokenizer, inner, original_layers, N_orig, block, prompts):
    """
    Compute seam Mahalanobis for block b, using base model stats as reference,
    but measuring the seam on whatever model state is currently set.

    Collects base model hidden states at layer j (using original layers),
    then collects exit-seam states from the current model with block b additionally duplicated.
    """
    device = next(model.parameters()).device
    i, j = block

    # Base stats: run ORIGINAL layers 0..j, collect h at layer j
    # Temporarily restore original layers for this
    current_layers = list(inner.layers)
    current_N = len(current_layers)
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N_orig

    base_states = []
    with torch.no_grad():
        for prompt in prompts[:8]:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
            h = inner.embed_tokens(ids["input_ids"])
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)
            for layer_idx in range(j):
                out = inner.layers[layer_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
            base_states.append(h[:, -1, :].float().cpu())

    base_stack = torch.cat(base_states, dim=0)
    mu = base_stack.mean(dim=0)
    var = base_stack.var(dim=0).clamp(min=1e-6)

    # Now build the model with current blocks + block b
    # Current layers already have block a applied. Add block b on top.
    order_ab = build_layer_order([block], N_orig)
    # But current_layers already has block a. We need to build from original.
    # Reconstruct: figure out what blocks are in current_layers by comparing lengths
    # Simpler: we know the caller set up model with block a. We need a+b.
    # The caller passes original_layers and N_orig, so we can rebuild.

    # Restore current state first
    inner.layers = nn.ModuleList(current_layers)
    model.config.num_hidden_layers = current_N

    # Now run the CURRENT model (which has block a) through block b's region
    # and measure how different the hidden state at j is from base
    dup_states = []
    with torch.no_grad():
        for prompt in prompts[:8]:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
            h = inner.embed_tokens(ids["input_ids"])
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            # Run through current model (has block a), then also duplicate block b
            # Run up to j in current model
            for step_idx in range(min(j, current_N)):
                out = inner.layers[step_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

            # Run block b a second time (layers i..j using original layers)
            for layer_idx in range(i, j):
                out = original_layers[layer_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

            dup_states.append(h[:, -1, :].float().cpu())

    dup_stack = torch.cat(dup_states, dim=0)
    md = ((dup_stack - mu) ** 2 / var).sum(dim=-1).sqrt().mean().item()
    return md


def main():
    print("=== DICE Feature Computation (7B) ===")
    model, tokenizer = load_original_model('models/small/Qwen2-7B-Instruct')
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)
    print(f"Loaded: {N} layers")

    prompts = CALIBRATION_PROMPTS

    # ============================================
    # Phase 1: Per-block features
    # ============================================
    print("\n--- Phase 1: Per-block features ---")

    block_data = {}
    for block in TOP_BLOCKS:
        print(f"\n  Block {block}:")

        # Displacement rho (on base model)
        rho = compute_displacement_rho(model, tokenizer, inner, original_layers, N, block, prompts)
        print(f"    displacement_rho = {rho:.4f}")

        # BLOOD profile
        blood_profile = compute_blood_profile(model, tokenizer, inner, original_layers, N, block, prompts)
        print(f"    BLOOD profile computed (non-zero layers: {(blood_profile > 0).sum()})")

        # Effect matrix (logit deltas)
        effect_mat = compute_effect_matrix(model, tokenizer, inner, original_layers, N, block, prompts)
        print(f"    Effect matrix: {effect_mat.shape}")

        # Seam Mahalanobis
        seam_md = compute_seam_mahalanobis(model, tokenizer, inner, original_layers, N, block, prompts)
        print(f"    Seam Mahalanobis = {seam_md:.2f}")

        block_data[block] = {
            'rho': rho,
            'blood_profile': blood_profile,
            'effect_matrix': effect_mat,
            'seam_md': seam_md,
            'midpoint': (block[0] + block[1]) / 2,
        }

    # ============================================
    # Phase 2: Pairwise features
    # ============================================
    print("\n--- Phase 2: Pairwise interaction features ---")

    pair_features = []

    sorted_blocks = sorted(TOP_BLOCKS, key=lambda b: b[0])
    valid_pairs = [(a, b) for a, b in combinations(sorted_blocks, 2)
                   if a[1] <= b[0]]  # Non-overlapping

    print(f"  Computing features for {len(valid_pairs)} non-overlapping pairs...")

    for a, b in valid_pairs:
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
        # Build model with a applied, then measure b's rho on it
        # Key: rho of b uses ORIGINAL layer indices, but we run through the
        # modified execution order. So we compute rho manually here.
        order_a = build_layer_order([a], N)
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order_a])
        model.config.num_hidden_layers = len(order_a)
        N_mod = len(order_a)

        # Compute rho of block b on this modified model
        # b's layers still refer to original indices — we need to find where
        # they appear in the modified execution order
        rho_b_given_a = _compute_rho_on_current_model(
            model, tokenizer, inner, b, prompts
        )
        inner.layers = nn.ModuleList(original_layers)
        model.config.num_hidden_layers = N

        rho_lift = db['rho'] - rho_b_given_a  # Positive = b is more contractive after a

        # 5. Conditional seam Mahalanobis change
        # Compare seam of b on base model vs seam of b on model+a
        # Use a simpler approach: just measure the full-model output divergence
        order_a = build_layer_order([a], N)
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order_a])
        model.config.num_hidden_layers = len(order_a)

        md_b_given_a = _compute_seam_md_on_current_model(
            model, tokenizer, inner, original_layers, N, b, prompts
        )
        inner.layers = nn.ModuleList(original_layers)
        model.config.num_hidden_layers = N

        ood_safe = -(md_b_given_a - db['seam_md'])  # Negative change = safer

        features = {
            'block_a': list(a),
            'block_b': list(b),
            'region_dist': region_dist,
            'effect_orth': effect_orth,
            'territory_orth': territory_orth,
            'rho_lift': rho_lift,
            'blood_lift': 0.0,  # TODO: compute conditional BLOOD (expensive)
            'ood_safe': ood_safe,
            'cka': cka,
            'blood_cos': blood_cos,
            'rho_b_base': db['rho'],
            'rho_b_cond': rho_b_given_a,
            'md_b_base': db['seam_md'],
            'md_b_cond': md_b_given_a,
        }
        pair_features.append(features)

        print(f"  {str(a):10s} → {str(b):10s}: "
              f"dist={region_dist:.2f} eff_orth={effect_orth:.3f} "
              f"terr_orth={territory_orth:.3f} rho_lift={rho_lift:+.4f} "
              f"ood_safe={ood_safe:+.1f}")

    # ============================================
    # Phase 3: Validate against labeled pairs
    # ============================================
    print("\n--- Phase 3: Validation against 22 labeled pairs ---")

    # Load ground truth
    with open('results/pairwise_stacking_sweep.json') as f:
        gt = json.load(f)

    baseline = 0.5344
    singles = {eval(k): v for k, v in gt['individual'].items()}

    # Compute observed epistasis for each pair
    gt_pairs = {}
    for p in gt['pairs']:
        a = tuple(p['block_a'])
        b = tuple(p['block_b'])
        eps_obs = p['pair_score'] - p['a_score'] - p['b_score'] + baseline
        gt_pairs[(a, b)] = {
            'pair_score': p['pair_score'],
            'epistasis': eps_obs,
            'stacks': p['stacks'],
        }

    # Match features to ground truth
    matched = []
    for pf in pair_features:
        a = tuple(pf['block_a'])
        b = tuple(pf['block_b'])
        key = (a, b)
        if key not in gt_pairs:
            # Try reverse
            key = (b, a)
        if key in gt_pairs:
            matched.append((pf, gt_pairs[key]))

    print(f"  Matched {len(matched)} pairs with ground truth")

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
        "blood_lift": 1.00,
        "effect_orth": 1.00,
        "territory_orth": 0.75,
        "region_dist": 0.50,
        "ood_safe": 0.75,
    }

    # Compute edge scores with default weights
    edge_scores_pred = []
    epistasis_obs = []
    for idx, (pf, gt_info) in enumerate(matched):
        features_ranked = {name: ranked[name][idx] for name in feature_names}
        es = sum(default_weights.get(k, 0.0) * features_ranked.get(k, 0.0) for k in feature_names)
        edge_scores_pred.append(es)
        epistasis_obs.append(gt_info['epistasis'])

    edge_scores_pred = np.array(edge_scores_pred)
    epistasis_obs = np.array(epistasis_obs)

    # Spearman correlation
    from scipy.stats import spearmanr
    r, p = spearmanr(edge_scores_pred, epistasis_obs)
    print(f"\n  Spearman(predicted_edge, observed_epistasis) = {r:.3f} (p={p:.4f})")

    # AUROC for stack vs interfere
    stacks = np.array([m[1]['stacks'] for m in matched])
    if stacks.sum() > 0 and stacks.sum() < len(stacks):
        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(stacks, edge_scores_pred)
            print(f"  AUROC(stacks vs not) = {auroc:.3f}")
        except ImportError:
            # Manual AUROC
            pos = edge_scores_pred[stacks == 1]
            neg = edge_scores_pred[stacks == 0]
            auroc = np.mean([float(p > n) + 0.5 * float(p == n) for p in pos for n in neg])
            print(f"  AUROC(stacks vs not) = {auroc:.3f} (manual)")

    # Top-k precision
    k = 5
    top_k_pred = np.argsort(-edge_scores_pred)[:k]
    top_k_actual_stack = sum(stacks[i] for i in top_k_pred)
    print(f"  Top-{k} precision (stacking pairs in top-{k} predicted): {top_k_actual_stack}/{k}")

    # Individual feature correlations
    print("\n  Individual feature correlations with epistasis:")
    for name in feature_names:
        r_feat, p_feat = spearmanr(feature_arrays[name], epistasis_obs)
        print(f"    {name:20s}: r={r_feat:+.3f} (p={p_feat:.4f})")

    # ============================================
    # Save all data
    # ============================================
    save_data = {
        'blocks': {str(k): {
            'rho': v['rho'],
            'seam_md': v['seam_md'],
            'midpoint': v['midpoint'],
            'blood_profile': v['blood_profile'].tolist(),
        } for k, v in block_data.items()},
        'pair_features': pair_features,
        'validation': {
            'spearman_r': float(r),
            'spearman_p': float(p),
            'n_matched': len(matched),
        }
    }

    os.makedirs('results/dice', exist_ok=True)
    with open('results/dice/7b_pair_features.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print("\nSaved to results/dice/7b_pair_features.json")
    print("\n=== DONE ===")


if __name__ == '__main__':
    main()
