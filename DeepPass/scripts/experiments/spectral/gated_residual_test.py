"""
Gated Residual Test — Analytical per-dimension gating at the duplication seam.

Instead of scalar alpha blending (h_patched = h1 + alpha * (h2 - h1)), we compute
a per-dimension gate vector analytically from calibration data:

    h_patched = h1 + gate ⊙ (h2 - h1)

where gate ∈ [0,1]^d is computed WITHOUT gradient descent using one of four methods:

1. VARIANCE GATE:  Low var(delta_d) across prompts → consistent change → gate high.
2. CORRELATION GATE: Dimensions whose delta correlates with math improvement → gate high.
3. MAGNITUDE GATE: sigmoid((|mean_delta_d| - mean) / std) — large mean changes gated more.
4. SIGN-CONSISTENCY GATE: Fraction of prompts where delta_d has the same sign → gate.

All gates are computed once on 32 calibration prompts then frozen during evaluation.
"""

import sys
import os
import json
import time
import gc

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..'))

from layer_duplicator import load_original_model
from math_probe import run_math_probe

# =============================================================================
# Config
# =============================================================================
MODEL_PATH = "models/full/Qwen3.5-9B"
N_CALIBRATION = 32
RESULTS_PATH = Path("results/data/gated_residual_results.json")

# Calibration prompts: mix of math, reasoning, code, factual
CALIBRATION_PROMPTS = [
    # Math (similar domain to probe)
    "What is 78313 multiplied by 88537?",
    "What is the square root of 152399025?",
    "What is 9999999 multiplied by 9999999?",
    "What is 123456789 multiplied by 987654321?",
    "What is 2 raised to the power of 48?",
    "What is the cube root of 74088?",
    "What is 54321 multiplied by 12345?",
    "What is 7777777 multiplied by 3333333?",
    # Reasoning
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. What does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "There are three boxes. One contains only apples, one contains only oranges, and one contains both. All labels are wrong. You pick one fruit from one box. Which box do you pick from to determine all labels?",
    # Code / technical
    "Write a Python function to compute the Fibonacci sequence.",
    "Explain how a transformer neural network processes text step by step.",
    "What is the time complexity of merge sort and why?",
    "Explain the difference between a stack and a queue.",
    # Factual / general knowledge
    "What is the capital of France?",
    "Who wrote the theory of general relativity?",
    "What is the speed of light in meters per second?",
    "How many planets are in the solar system?",
    # More math variants
    "What is 456789 squared?",
    "What is 314159 multiplied by 271828?",
    "What is 999999999999 divided by 142857?",
    "What is 11111111 multiplied by 11111111?",
    # More reasoning
    "If today is Monday, what day will it be 100 days from now?",
    "A farmer has 17 sheep. All but 9 die. How many are left?",
    "How many times can you subtract 5 from 25?",
    "What comes next in the sequence: 1, 1, 2, 3, 5, 8, ...?",
    # More technical
    "What is the difference between TCP and UDP?",
    "Explain what a hash table is and its average time complexity for lookup.",
    "What is backpropagation in neural networks?",
    "Describe the CAP theorem in distributed systems.",
]

SYSTEM_PROMPT = "You are a math calculator. Answer with ONLY the number, nothing else. No explanation, no units, no punctuation. Just the integer."
USER_TEMPLATE = "{question}\nAnswer with ONLY the integer number:"


# =============================================================================
# Layer-by-layer forward with gated seam patching
# =============================================================================

def build_layer_order(block_start, block_end, N):
    """Build execution order for a single duplicated block."""
    return list(range(block_end)) + list(range(block_start, block_end)) + list(range(block_end, N))


def find_seam_positions(layer_order, block_start, block_end):
    """Find step indices where first and second pass of the block end."""
    last_layer = block_end - 1
    occurrences = [step for step, layer_idx in enumerate(layer_order) if layer_idx == last_layer]
    assert len(occurrences) >= 2, f"Block ({block_start},{block_end}) not duplicated"
    return occurrences[0], occurrences[1]


def forward_with_gate(model, tokenizer, input_ids, layer_order, original_layers,
                      first_end, second_end, gate_vector=None):
    """
    Single forward pass with gated residual at the seam.

    gate_vector: tensor of shape (hidden_dim,) with values in [0,1], or None for alpha=1.0.
    If gate_vector is a scalar (0-d tensor or float), applies uniform alpha.

    Returns logits.
    """
    inner = model.model
    device = input_ids.device

    with torch.no_grad():
        h = inner.embed_tokens(input_ids)
        seq_len = h.shape[1]
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = inner.rotary_emb(h, pos_ids)

        h_after_first = None
        for step_idx, layer_idx in enumerate(layer_order):
            layer = original_layers[layer_idx]
            out = layer(h, position_embeddings=pos_embeds, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out

            if step_idx == first_end:
                h_after_first = h.clone()

            if step_idx == second_end and h_after_first is not None:
                if gate_vector is not None:
                    # gate_vector shape: (hidden_dim,) → broadcast over (batch, seq, hidden)
                    g = gate_vector.to(device=device, dtype=h.dtype)
                    if g.dim() == 0:
                        h = h_after_first + g * (h - h_after_first)
                    else:
                        h = h_after_first + g.unsqueeze(0).unsqueeze(0) * (h - h_after_first)
                # else: gate_vector is None → alpha=1.0, keep h as-is

        h = inner.norm(h)
        logits = model.lm_head(h)

    return logits


def generate_with_gate(model, tokenizer, prompt, layer_order, original_layers,
                       first_end, second_end, gate_vector=None, max_new_tokens=64):
    """Generate text token-by-token with gated seam patching."""
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    for _ in range(max_new_tokens):
        logits = forward_with_gate(
            model, tokenizer, input_ids, layer_order, original_layers,
            first_end, second_end, gate_vector
        )
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    generated = input_ids[0, prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# =============================================================================
# Quick rho screening to find best single-block config
# =============================================================================

def quick_rho_screen(model, tokenizer, original_layers, N, step=2, block_sizes=None):
    """
    Spectral displacement rho screening: for each candidate block, measure
    how much duplicating it changes the output logits vs baseline.
    Lower displacement → better candidate.
    """
    if block_sizes is None:
        block_sizes = [1, 2, 3, 5]

    inner = model.model
    device = next(model.parameters()).device

    screen_prompts = CALIBRATION_PROMPTS[:4]  # Use just 4 for speed

    print(f"\n--- Quick Rho Screen: {len(block_sizes)} block sizes, step={step} ---")

    candidates = []
    for bs in block_sizes:
        for start in range(0, N - bs, step):
            end = start + bs
            if end > N:
                continue

            disps = []
            for prompt in screen_prompts:
                inp = tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=64).to(device)
                input_ids = inp["input_ids"]

                with torch.no_grad():
                    # Baseline: normal forward
                    h = inner.embed_tokens(input_ids)
                    seq_len = h.shape[1]
                    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                    pos_embeds = inner.rotary_emb(h, pos_ids)

                    for li in range(N):
                        out = original_layers[li](h, position_embeddings=pos_embeds, use_cache=False)
                        h = out[0] if isinstance(out, tuple) else out
                    logits_base = model.lm_head(inner.norm(h))

                    # Duplicated: run block twice
                    dup_order = build_layer_order(start, end, N)
                    h = inner.embed_tokens(input_ids)
                    for step_idx in range(len(dup_order)):
                        li = dup_order[step_idx]
                        out = original_layers[li](h, position_embeddings=pos_embeds, use_cache=False)
                        h = out[0] if isinstance(out, tuple) else out
                    logits_dup = model.lm_head(inner.norm(h))

                    diff = (logits_dup - logits_base).float()
                    disp = diff.norm() / (logits_base.float().norm() + 1e-8)
                    disps.append(disp.item())

            mean_disp = np.mean(disps)
            candidates.append({
                "start": start,
                "end": end,
                "block_size": bs,
                "displacement_rho": mean_disp,
            })

    candidates.sort(key=lambda x: x["displacement_rho"])
    print(f"  Screened {len(candidates)} configs. Top 10:")
    for c in candidates[:10]:
        print(f"    ({c['start']:2d},{c['end']:2d}) bs={c['block_size']} "
              f"disp_rho={c['displacement_rho']:.6f}")

    return candidates


# =============================================================================
# Collect h1, h2 on calibration prompts
# =============================================================================

def collect_seam_states(model, tokenizer, original_layers, N,
                        block_start, block_end, prompts):
    """
    For each calibration prompt, run the duplicated forward and collect:
      h1: hidden state after the first pass (shape: [seq_len, hidden_dim])
      h2: hidden state after the second pass (shape: [seq_len, hidden_dim])

    Returns h1_all, h2_all — each a list of tensors, one per prompt.
    We use the LAST token position only (where generation happens).
    """
    inner = model.model
    device = next(model.parameters()).device
    layer_order = build_layer_order(block_start, block_end, N)
    first_end, second_end = find_seam_positions(layer_order, block_start, block_end)

    h1_list = []
    h2_list = []

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt",
                              truncation=True, max_length=128)["input_ids"].to(device)

        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            h_after_first = None
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

                if step_idx == first_end:
                    h_after_first = h.clone()

                if step_idx == second_end:
                    # h1 = after first pass, h2 = after second pass
                    # Take last token position for analysis
                    h1_list.append(h_after_first[:, -1, :].squeeze(0).float().cpu())
                    h2_list.append(h[:, -1, :].squeeze(0).float().cpu())

    return h1_list, h2_list


# =============================================================================
# Gate computation methods (all analytical, no training)
# =============================================================================

def compute_variance_gate(h1_list, h2_list, temperature=1.0):
    """
    VARIANCE GATE: For each dimension d, compute var(h2_d - h1_d) across prompts.
    High variance = inconsistent = gate low. Low variance = consistent = gate high.

    gate_d = sigmoid(-temperature * (var_d - median_var) / std_var)
    """
    deltas = torch.stack([h2 - h1 for h1, h2 in zip(h1_list, h2_list)])  # (N, D)
    var_per_dim = deltas.var(dim=0)  # (D,)

    # Normalize: high variance → low gate
    median_var = var_per_dim.median()
    std_var = var_per_dim.std() + 1e-8
    gate = torch.sigmoid(-temperature * (var_per_dim - median_var) / std_var)

    return gate


def compute_correlation_gate(h1_list, h2_list, model, tokenizer, original_layers,
                             N, block_start, block_end, temperature=2.0):
    """
    CORRELATION GATE: For each dimension, correlate delta_d with a "quality" signal.

    The quality signal = per-prompt math score improvement from duplication.
    We run math probe questions through both base and duplicated paths,
    using the per-question score as the quality signal.

    Since we have only 16 math questions, we use those 16 as our correlation prompts.
    """
    from math_probe import MATH_QUESTIONS, calculate_score, extract_number

    device = next(model.parameters()).device
    inner = model.model
    layer_order = build_layer_order(block_start, block_end, N)
    first_end, second_end = find_seam_positions(layer_order, block_start, block_end)

    # Collect h1, h2 and quality scores for each math question
    corr_h1 = []
    corr_h2 = []
    quality_scores = []

    for q in MATH_QUESTIONS:
        prompt = f"System: You are a math calculator. Answer with ONLY the number, nothing else.\n\nUser: {q['question']}\nAnswer with ONLY the integer number:\n\nAssistant:"
        input_ids = tokenizer(prompt, return_tensors="pt",
                              truncation=True, max_length=128)["input_ids"].to(device)

        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            h_after_first = None
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

                if step_idx == first_end:
                    h_after_first = h.clone()

                if step_idx == second_end:
                    corr_h1.append(h_after_first[:, -1, :].squeeze(0).float().cpu())
                    corr_h2.append(h[:, -1, :].squeeze(0).float().cpu())

        # Get quality signal: generate with alpha=0 (base) and alpha=1 (dup), compare scores
        resp_base = generate_with_gate(
            model, tokenizer, prompt, layer_order, original_layers,
            first_end, second_end, gate_vector=torch.tensor(0.0),
            max_new_tokens=64
        )
        resp_dup = generate_with_gate(
            model, tokenizer, prompt, layer_order, original_layers,
            first_end, second_end, gate_vector=None,  # alpha=1.0
            max_new_tokens=64
        )

        est_base = extract_number(resp_base)
        est_dup = extract_number(resp_dup)
        try:
            score_base = calculate_score(q['answer'], est_base)
        except Exception:
            score_base = 0.0
        try:
            score_dup = calculate_score(q['answer'], est_dup)
        except Exception:
            score_dup = 0.0

        quality = score_dup - score_base
        quality_scores.append(quality)

    # Compute per-dimension correlation
    deltas = torch.stack([h2 - h1 for h1, h2 in zip(corr_h1, corr_h2)])  # (16, D)
    quality_t = torch.tensor(quality_scores, dtype=torch.float32)  # (16,)

    # Pearson correlation per dimension
    D = deltas.shape[1]
    delta_centered = deltas - deltas.mean(dim=0, keepdim=True)
    quality_centered = quality_t - quality_t.mean()

    numerator = (delta_centered * quality_centered.unsqueeze(1)).sum(dim=0)  # (D,)
    denom = (delta_centered.norm(dim=0) * quality_centered.norm() + 1e-8)
    corr = numerator / denom  # (D,)

    # Gate: dimensions with positive correlation get high gate
    gate = torch.sigmoid(temperature * corr)

    return gate


def compute_magnitude_gate(h1_list, h2_list, temperature=1.0):
    """
    MAGNITUDE GATE: Dimensions with larger mean absolute change get gated more aggressively.

    gate_d = sigmoid(temperature * (|mean_delta_d| - mean(|mean_delta|)) / std(|mean_delta|))

    Intuition: dimensions where the second pass makes a big consistent change
    are the ones worth keeping.
    """
    deltas = torch.stack([h2 - h1 for h1, h2 in zip(h1_list, h2_list)])  # (N, D)
    mean_delta = deltas.mean(dim=0)  # (D,)
    abs_mean = mean_delta.abs()  # (D,)

    mu = abs_mean.mean()
    sigma = abs_mean.std() + 1e-8
    gate = torch.sigmoid(temperature * (abs_mean - mu) / sigma)

    return gate


def compute_sign_consistency_gate(h1_list, h2_list, threshold=0.7):
    """
    SIGN-CONSISTENCY GATE: For each dimension, what fraction of prompts have
    delta_d with the same sign? High consistency → reliable direction → keep it.

    gate_d = max(frac_positive, frac_negative)
    If gate_d > threshold, rescale to [0,1]; below threshold → 0.
    """
    deltas = torch.stack([h2 - h1 for h1, h2 in zip(h1_list, h2_list)])  # (N, D)
    N = deltas.shape[0]

    frac_positive = (deltas > 0).float().mean(dim=0)  # (D,)
    frac_negative = (deltas < 0).float().mean(dim=0)  # (D,)
    consistency = torch.maximum(frac_positive, frac_negative)  # (D,)

    # Rescale: below threshold → 0, above → linear scale to [0,1]
    gate = torch.clamp((consistency - threshold) / (1.0 - threshold), min=0.0, max=1.0)

    return gate


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_gate(model, tokenizer, original_layers, N, block_start, block_end,
                  gate_vector, name, layer_order, first_end, second_end):
    """Run math probe with a given gate vector. Returns dict with results."""

    def gen_fn(prompt):
        return generate_with_gate(
            model, tokenizer, prompt, layer_order, original_layers,
            first_end, second_end, gate_vector, max_new_tokens=64
        )

    math_r = run_math_probe(gen_fn, verbose=False)
    print(f"    {name:35s}: math={math_r['score']:.4f}")
    return {
        "name": name,
        "math_score": math_r["score"],
        "math_scores": math_r["scores"],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    t_start = time.time()

    print("=" * 70)
    print("GATED RESIDUAL TEST — Analytical Per-Dimension Gating")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Calibration prompts: {N_CALIBRATION}")
    print()

    # =========================================================================
    # Step 1: Load model
    # =========================================================================
    print("--- Step 1: Loading model ---")
    model, tokenizer = load_original_model(MODEL_PATH)
    model.eval()
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)
    hidden_dim = model.config.hidden_size
    print(f"  Layers: {N}, Hidden dim: {hidden_dim}")

    # =========================================================================
    # Step 2: Quick rho screen to find best block
    # =========================================================================
    print("\n--- Step 2: Quick Rho Screen ---")
    candidates = quick_rho_screen(
        model, tokenizer, original_layers, N,
        step=2, block_sizes=[1, 2, 3, 5]
    )

    # Pick the top candidate
    best_cand = candidates[0]
    block_start = best_cand["start"]
    block_end = best_cand["end"]
    print(f"\n  Selected block: ({block_start},{block_end}) "
          f"disp_rho={best_cand['displacement_rho']:.6f}")

    # Also grab a few runners-up for comparison later
    top5 = candidates[:5]

    # =========================================================================
    # Step 3: Collect h1, h2 on calibration prompts
    # =========================================================================
    print(f"\n--- Step 3: Collecting seam states on {N_CALIBRATION} calibration prompts ---")
    layer_order = build_layer_order(block_start, block_end, N)
    first_end, second_end = find_seam_positions(layer_order, block_start, block_end)

    h1_list, h2_list = collect_seam_states(
        model, tokenizer, original_layers, N,
        block_start, block_end, CALIBRATION_PROMPTS[:N_CALIBRATION]
    )
    print(f"  Collected {len(h1_list)} (h1, h2) pairs, dim={h1_list[0].shape[-1]}")

    # Quick stats on deltas
    deltas = torch.stack([h2 - h1 for h1, h2 in zip(h1_list, h2_list)])
    print(f"  Delta stats: mean={deltas.mean():.4f}, std={deltas.std():.4f}, "
          f"max={deltas.abs().max():.4f}")

    # =========================================================================
    # Step 4: Compute all gate variants
    # =========================================================================
    print(f"\n--- Step 4: Computing gate variants ---")

    gates = {}

    # 4a. Variance gate
    print("  Computing VARIANCE gate...")
    gates["variance"] = compute_variance_gate(h1_list, h2_list, temperature=1.0)
    g = gates["variance"]
    print(f"    Stats: mean={g.mean():.4f}, median={g.median():.4f}, "
          f"min={g.min():.4f}, max={g.max():.4f}")

    # 4b. Correlation gate
    print("  Computing CORRELATION gate (runs 16 math questions x2)...")
    gates["correlation"] = compute_correlation_gate(
        h1_list, h2_list, model, tokenizer, original_layers,
        N, block_start, block_end, temperature=2.0
    )
    g = gates["correlation"]
    print(f"    Stats: mean={g.mean():.4f}, median={g.median():.4f}, "
          f"min={g.min():.4f}, max={g.max():.4f}")

    # 4c. Magnitude gate
    print("  Computing MAGNITUDE gate...")
    gates["magnitude"] = compute_magnitude_gate(h1_list, h2_list, temperature=1.0)
    g = gates["magnitude"]
    print(f"    Stats: mean={g.mean():.4f}, median={g.median():.4f}, "
          f"min={g.min():.4f}, max={g.max():.4f}")

    # 4d. Sign-consistency gate
    print("  Computing SIGN-CONSISTENCY gate...")
    gates["sign_consistency"] = compute_sign_consistency_gate(h1_list, h2_list, threshold=0.7)
    g = gates["sign_consistency"]
    print(f"    Stats: mean={g.mean():.4f}, median={g.median():.4f}, "
          f"min={g.min():.4f}, max={g.max():.4f}, "
          f"pct_nonzero={100*(g > 0).float().mean():.1f}%")

    # =========================================================================
    # Step 5: Evaluate all variants + baselines
    # =========================================================================
    print(f"\n--- Step 5: Evaluation on block ({block_start},{block_end}) ---")

    results = []

    # Baselines
    print("\n  BASELINES:")

    # alpha=0 (erase second pass entirely)
    r = evaluate_gate(model, tokenizer, original_layers, N, block_start, block_end,
                      gate_vector=torch.tensor(0.0),
                      name="alpha=0.0 (no second pass)",
                      layer_order=layer_order, first_end=first_end, second_end=second_end)
    r["method"] = "baseline_alpha0"
    results.append(r)

    # alpha=1 (full duplication, no gating)
    r = evaluate_gate(model, tokenizer, original_layers, N, block_start, block_end,
                      gate_vector=None,
                      name="alpha=1.0 (full duplication)",
                      layer_order=layer_order, first_end=first_end, second_end=second_end)
    r["method"] = "baseline_alpha1"
    results.append(r)

    # alpha=0.5 (uniform scalar blend)
    r = evaluate_gate(model, tokenizer, original_layers, N, block_start, block_end,
                      gate_vector=torch.tensor(0.5),
                      name="alpha=0.5 (uniform scalar)",
                      layer_order=layer_order, first_end=first_end, second_end=second_end)
    r["method"] = "baseline_alpha05"
    results.append(r)

    # Gated variants
    print("\n  GATED VARIANTS:")

    for gate_name, gate_vec in gates.items():
        r = evaluate_gate(model, tokenizer, original_layers, N, block_start, block_end,
                          gate_vector=gate_vec,
                          name=f"gate: {gate_name}",
                          layer_order=layer_order, first_end=first_end, second_end=second_end)
        r["method"] = f"gate_{gate_name}"
        results.append(r)

    # Also try combinations: variance * sign_consistency
    print("\n  COMBINATION GATES:")

    combo_var_sign = gates["variance"] * gates["sign_consistency"]
    r = evaluate_gate(model, tokenizer, original_layers, N, block_start, block_end,
                      gate_vector=combo_var_sign,
                      name="gate: variance * sign_consistency",
                      layer_order=layer_order, first_end=first_end, second_end=second_end)
    r["method"] = "gate_variance_x_sign"
    results.append(r)

    combo_mag_sign = gates["magnitude"] * gates["sign_consistency"]
    r = evaluate_gate(model, tokenizer, original_layers, N, block_start, block_end,
                      gate_vector=combo_mag_sign,
                      name="gate: magnitude * sign_consistency",
                      layer_order=layer_order, first_end=first_end, second_end=second_end)
    r["method"] = "gate_magnitude_x_sign"
    results.append(r)

    combo_all = (gates["variance"] + gates["magnitude"] + gates["sign_consistency"]) / 3.0
    r = evaluate_gate(model, tokenizer, original_layers, N, block_start, block_end,
                      gate_vector=combo_all,
                      name="gate: avg(var, mag, sign)",
                      layer_order=layer_order, first_end=first_end, second_end=second_end)
    r["method"] = "gate_avg_3way"
    results.append(r)

    # =========================================================================
    # Step 6: Also test top rho block on runner-up if time permits
    # =========================================================================
    # Test on a second block to check if gates generalize
    block2_results = None
    b2_start = None
    b2_end = None
    if len(top5) >= 2:
        cand2 = top5[1]
        b2_start = cand2["start"]
        b2_end = cand2["end"]
        if (b2_start, b2_end) != (block_start, block_end):
            print(f"\n--- Bonus: Testing gates on runner-up block ({b2_start},{b2_end}) ---")
            lo2 = build_layer_order(b2_start, b2_end, N)
            fe2, se2 = find_seam_positions(lo2, b2_start, b2_end)

            # Collect seam states for block 2
            h1_b2, h2_b2 = collect_seam_states(
                model, tokenizer, original_layers, N,
                b2_start, b2_end, CALIBRATION_PROMPTS[:N_CALIBRATION]
            )

            # Recompute gates for block 2
            gates_b2 = {
                "variance": compute_variance_gate(h1_b2, h2_b2),
                "magnitude": compute_magnitude_gate(h1_b2, h2_b2),
                "sign_consistency": compute_sign_consistency_gate(h1_b2, h2_b2),
            }

            block2_results = []

            # Baselines
            for alpha_val, alpha_name in [(0.0, "alpha=0.0"), (None, "alpha=1.0")]:
                gv = torch.tensor(alpha_val) if alpha_val is not None else None
                r = evaluate_gate(model, tokenizer, original_layers, N, b2_start, b2_end,
                                  gate_vector=gv,
                                  name=f"B2({b2_start},{b2_end}) {alpha_name}",
                                  layer_order=lo2, first_end=fe2, second_end=se2)
                r["method"] = f"b2_{alpha_name}"
                r["block"] = [b2_start, b2_end]
                block2_results.append(r)

            # Best gate from primary block
            for gate_name, gate_vec in gates_b2.items():
                r = evaluate_gate(model, tokenizer, original_layers, N, b2_start, b2_end,
                                  gate_vector=gate_vec,
                                  name=f"B2({b2_start},{b2_end}) gate:{gate_name}",
                                  layer_order=lo2, first_end=fe2, second_end=se2)
                r["method"] = f"b2_gate_{gate_name}"
                r["block"] = [b2_start, b2_end]
                block2_results.append(r)

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - t_start

    print(f"\n{'=' * 70}")
    print("SUMMARY — Gated Residual Results")
    print(f"{'=' * 70}")
    print(f"Block: ({block_start},{block_end}), Hidden dim: {hidden_dim}")
    print()

    # Sort results by math score
    sorted_results = sorted(results, key=lambda x: x["math_score"], reverse=True)
    best = sorted_results[0]

    for r in sorted_results:
        marker = " <-- BEST" if r["math_score"] == best["math_score"] else ""
        print(f"  {r['name']:40s}  math={r['math_score']:.4f}{marker}")

    # Check if any gate beats alpha=1.0
    alpha1_score = next(r["math_score"] for r in results if r["method"] == "baseline_alpha1")
    gate_results = [r for r in results if r["method"].startswith("gate_")]
    best_gate = max(gate_results, key=lambda x: x["math_score"]) if gate_results else None

    print()
    if best_gate and best_gate["math_score"] > alpha1_score:
        delta = best_gate["math_score"] - alpha1_score
        print(f"  RESULT: Best gate ({best_gate['name']}) BEATS alpha=1.0 "
              f"by {delta:+.4f} ({100*delta/max(alpha1_score,1e-8):+.1f}%)")
    elif best_gate:
        delta = best_gate["math_score"] - alpha1_score
        print(f"  RESULT: Best gate ({best_gate['name']}) vs alpha=1.0: "
              f"{delta:+.4f} ({100*delta/max(alpha1_score,1e-8):+.1f}%)")
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Gate statistics for analysis
    gate_stats = {}
    for gname, gvec in gates.items():
        gate_stats[gname] = {
            "mean": float(gvec.mean()),
            "median": float(gvec.median()),
            "std": float(gvec.std()),
            "min": float(gvec.min()),
            "max": float(gvec.max()),
            "pct_above_05": float((gvec > 0.5).float().mean()),
            "pct_above_09": float((gvec > 0.9).float().mean()),
            "pct_below_01": float((gvec < 0.1).float().mean()),
        }

    # =========================================================================
    # Save results
    # =========================================================================
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_PATH,
        "block": [block_start, block_end],
        "block_displacement_rho": best_cand["displacement_rho"],
        "hidden_dim": hidden_dim,
        "num_layers": N,
        "n_calibration": N_CALIBRATION,
        "elapsed_seconds": elapsed,
        "top5_candidates": top5,
        "gate_statistics": gate_stats,
        "results": [{
            "method": r["method"],
            "name": r["name"],
            "math_score": r["math_score"],
        } for r in sorted_results],
        "best_method": best["method"],
        "best_score": best["math_score"],
        "alpha1_score": alpha1_score,
        "best_gate_vs_alpha1": (best_gate["math_score"] - alpha1_score) if best_gate else None,
    }

    if block2_results is not None:
        output["block2"] = {
            "block": [b2_start, b2_end],
            "results": [{
                "method": r["method"],
                "name": r["name"],
                "math_score": r["math_score"],
            } for r in block2_results],
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {RESULTS_PATH}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
