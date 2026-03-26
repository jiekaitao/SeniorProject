"""
SVD Subspace Patching — Low-rank decomposition of the duplication residual.

Scalar alpha blending at the seam (h1 + alpha*(h2-h1)) always lands on alpha=1.0,
meaning the linear interpolation cannot improve on full duplication. But the residual
h2-h1 is a matrix (tokens x hidden_dim), and different directions of change may help
or hurt. This script decomposes h2-h1 via SVD and tests:

  1. Top-k patching:   keep only the top-k principal directions of change, zero the rest.
  2. Sign reversal:    reverse the sign on the bottom directions (the "harmful" ones).

Protocol:
  - Find the best single-block config via displacement rho (step=2).
  - Collect h1 (after first pass) and h2 (after second pass) on 32 calibration prompts.
  - SVD the residual matrix to get principal components.
  - For each variant, generate with the patched residual and evaluate with math_probe.
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

# -- imports from project scripts --
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, '/blue/cis4914/jietao/DeepPass/scripts')

from layer_duplicator import load_original_model
from math_probe import run_math_probe

# =============================================================================
# Config
# =============================================================================
MODEL_PATH = "/blue/cis4914/jietao/DeepPass/models/full/Qwen3.5-9B"
RESULTS_PATH = "/blue/cis4914/jietao/DeepPass/results/data/svd_subspace_patching.json"

# Calibration prompts for collecting h1/h2 residuals
CALIBRATION_PROMPTS = [
    "The theory of general relativity describes",
    "In mathematics, a topological space is",
    "def fibonacci(n):\n    if n <= 1:",
    "What is 123 multiplied by 456?",
    "The mitochondria is the powerhouse of",
    "According to quantum mechanics, particles",
    "Once upon a time in a land far away",
    "A linked list is a data structure where",
    "The Fourier transform decomposes a function into",
    "In organic chemistry, a benzene ring has",
    "The halting problem proves that no algorithm can",
    "Newton's second law states that force equals",
    "A red-black tree is a self-balancing binary",
    "The speed of light in vacuum is approximately",
    "In economics, the law of supply and demand",
    "The double helix structure of DNA was discovered by",
    "A convolutional neural network processes images by",
    "The second law of thermodynamics states that entropy",
    "In linear algebra, an eigenvector of a matrix",
    "The Pythagorean theorem says that in a right triangle",
    "Machine learning models can overfit when they memorize",
    "The human genome contains approximately three billion",
    "A hash table provides average-case constant time",
    "The Schrodinger equation describes how the quantum",
    "In graph theory, Euler's formula states that",
    "The central limit theorem tells us that the sum",
    "A compiler translates source code into machine",
    "The Drake equation estimates the number of",
    "Gradient descent is an optimization algorithm that",
    "The Heisenberg uncertainty principle states that",
    "In probability theory, Bayes' theorem relates",
    "The traveling salesman problem asks for the shortest",
]

# Spectral screening prompts (smaller set for speed)
SPECTRAL_PROMPTS = [
    "The theory of general relativity states that",
    "In Python, a decorator is a function that",
    "What is 78313 multiplied by 88537?",
    "A linked list is a data structure where",
]

TOP_K_VALUES = [1, 2, 4, 8, 16, 32]


# =============================================================================
# Step 1: Find the best single-layer block via displacement rho
# =============================================================================

def find_best_block_by_displacement(model, tokenizer, step=2):
    """
    Compute displacement rho for single-layer duplication at each block (step=2).
    The block with the LOWEST displacement rho is closest to a fixed point and
    most likely to benefit from duplication.
    Returns (best_start, best_end, all_candidates).
    """
    device = next(model.parameters()).device
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)

    print(f"\n{'='*70}")
    print(f"STEP 1: Displacement rho screening (N={N}, step={step})")
    print(f"{'='*70}")

    candidates = []
    for start in range(0, N, step):
        end = start + 1
        if end > N:
            continue

        disps = []
        for prompt in SPECTRAL_PROMPTS:
            inp = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=64).to(device)
            with torch.no_grad():
                # Base forward (no duplication)
                h = inner.embed_tokens(inp["input_ids"])
                seq_len = h.shape[1]
                pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                pos_embeds = inner.rotary_emb(h, pos_ids)

                h_base = h.clone()
                for i in range(N):
                    out = original_layers[i](h_base, position_embeddings=pos_embeds, use_cache=False)
                    h_base = out[0] if isinstance(out, tuple) else out
                logits_base = model.lm_head(inner.norm(h_base))

                # Duplicated forward: run layer[start] twice
                h_dup = h.clone()
                dup_order = list(range(end)) + list(range(start, end)) + list(range(end, N))
                for idx in dup_order:
                    out = original_layers[idx](h_dup, position_embeddings=pos_embeds, use_cache=False)
                    h_dup = out[0] if isinstance(out, tuple) else out
                logits_dup = model.lm_head(inner.norm(h_dup))

                diff = (logits_dup - logits_base).float()
                disp = diff.norm() / (logits_base.float().norm() + 1e-8)
                disps.append(disp.item())

        avg_disp = np.mean(disps)
        candidates.append({
            "start": start, "end": end,
            "displacement_rho": avg_disp,
        })
        print(f"  Block ({start:3d},{end:3d}): disp_rho = {avg_disp:.6f}")

    # Sort by displacement rho (lower = more fixed-point-like = better for duplication)
    candidates.sort(key=lambda x: x["displacement_rho"])

    print(f"\n  Top 5 candidates (lowest displacement rho):")
    for i, c in enumerate(candidates[:5]):
        print(f"    {i+1}. ({c['start']},{c['end']}) disp_rho={c['displacement_rho']:.6f}")

    best = candidates[0]
    print(f"\n  --> Best block: ({best['start']},{best['end']}) "
          f"disp_rho={best['displacement_rho']:.6f}")

    return best["start"], best["end"], candidates


# =============================================================================
# Step 2: Collect h1 and h2 hidden states at the seam
# =============================================================================

def build_layer_order(block_start, block_end, N):
    """Build execution order with one block duplicated."""
    return list(range(block_end)) + list(range(block_start, block_end)) + list(range(block_end, N))


def find_seam_positions(layer_order, block_start, block_end):
    """
    Find step indices where first and second pass of the block end.
    Returns (first_pass_end_step, second_pass_end_step).
    """
    last_layer = block_end - 1
    occurrences = [step for step, layer_idx in enumerate(layer_order) if layer_idx == last_layer]
    assert len(occurrences) >= 2, (
        f"Block ({block_start},{block_end}) not duplicated. "
        f"Layer {last_layer} appears {len(occurrences)} time(s)"
    )
    return occurrences[0], occurrences[1]


def collect_residuals(model, tokenizer, block_start, block_end, prompts):
    """
    Run the model with block duplication and collect h1, h2, and (h2-h1) at the seam.

    Returns:
        residuals: list of (h2-h1) tensors, each shape (seq_len, hidden_dim) in float32
        h1_list:   list of h1 tensors
        h2_list:   list of h2 tensors
    """
    device = next(model.parameters()).device
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)

    layer_order = build_layer_order(block_start, block_end, N)
    first_end, second_end = find_seam_positions(layer_order, block_start, block_end)

    print(f"\n{'='*70}")
    print(f"STEP 2: Collecting h1/h2 residuals at seam of block ({block_start},{block_end})")
    print(f"{'='*70}")
    print(f"  Layer order length: {len(layer_order)}")
    print(f"  First pass ends at step {first_end}, second at step {second_end}")
    print(f"  Collecting on {len(prompts)} calibration prompts...")

    residuals = []
    h1_list = []
    h2_list = []

    for pidx, prompt in enumerate(prompts):
        inp = tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=64).to(device)
        with torch.no_grad():
            h = inner.embed_tokens(inp["input_ids"])
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            h_after_first = None
            h_after_second = None
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

                if step_idx == first_end:
                    h_after_first = h.clone()

                if step_idx == second_end:
                    h_after_second = h.clone()

            # h1 and h2 captured at the seam (not end of full forward)
            h1 = h_after_first[0].float().cpu()  # (seq_len, hidden_dim)
            h2 = h_after_second[0].float().cpu()  # (seq_len, hidden_dim)
            residual = h2 - h1                     # (seq_len, hidden_dim)

            residuals.append(residual)
            h1_list.append(h1)
            h2_list.append(h2)

        if (pidx + 1) % 8 == 0:
            print(f"    Collected {pidx+1}/{len(prompts)}")

    print(f"  Done. Residual shapes: {residuals[0].shape} each")
    return residuals, h1_list, h2_list


# =============================================================================
# Step 3: SVD decomposition of residuals
# =============================================================================

def compute_svd(residuals):
    """
    Compute SVD of the stacked residual matrix.

    Stack all (h2-h1) from all prompts into one big matrix (total_tokens x hidden_dim),
    then decompose: R = U @ diag(S) @ Vh

    Returns:
        U:  (total_tokens, min(total_tokens, hidden_dim))
        S:  (min(total_tokens, hidden_dim),)
        Vh: (min(total_tokens, hidden_dim), hidden_dim)  <-- the directions in hidden space
    """
    print(f"\n{'='*70}")
    print(f"STEP 3: SVD decomposition of residuals")
    print(f"{'='*70}")

    # Stack all residuals: (total_tokens, hidden_dim)
    R = torch.cat(residuals, dim=0)  # (total_tokens, hidden_dim)
    print(f"  Stacked residual matrix: {R.shape}")

    t0 = time.time()
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    elapsed = time.time() - t0
    print(f"  SVD computed in {elapsed:.1f}s")
    print(f"  U: {U.shape}, S: {S.shape}, Vh: {Vh.shape}")

    # Report variance explained by top components
    total_var = (S ** 2).sum().item()
    for k in [1, 2, 4, 8, 16, 32, 64]:
        if k <= len(S):
            var_k = (S[:k] ** 2).sum().item()
            pct = 100.0 * var_k / total_var
            print(f"    Top-{k:3d}: {pct:6.2f}% variance explained")

    return U, S, Vh


# =============================================================================
# Step 4: Generate text with SVD-patched residual
# =============================================================================

def generate_with_svd_patch(model, tokenizer, prompt, layer_order, original_layers,
                             block_start, block_end, first_end, second_end,
                             Vh, S, patch_mode, patch_k, max_new_tokens=64):
    """
    Generate text token-by-token with manual layer-by-layer forward.
    At the seam, instead of using the raw residual (h2-h1), project it through
    a modified SVD basis.

    patch_mode:
        "topk"        — keep only top-k directions, zero the rest
        "zero_bottom"  — zero the bottom (total-k) directions, keep top-k (same as topk)
        "reverse_bottom" — keep top-k, REVERSE sign on bottom directions
        "full"         — alpha=1.0, no patching (baseline: full duplication)
        "none"         — alpha=0.0, no duplication effect at seam (equivalent to h1)
    """
    inner = model.model
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # Precompute projection matrices on the correct device and dtype
    Vh_dev = Vh.to(device)  # (num_components, hidden_dim) in float32
    n_components = Vh_dev.shape[0]

    for _ in range(max_new_tokens):
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

                # Cache h1 (after first pass)
                if step_idx == first_end:
                    h_after_first = h.clone()

                # Apply SVD patch after second pass
                if step_idx == second_end and h_after_first is not None:
                    if patch_mode == "full":
                        # No patching — keep h2 as-is (alpha=1.0)
                        pass
                    elif patch_mode == "none":
                        # Revert to h1 (alpha=0.0)
                        h = h_after_first
                    else:
                        # Compute residual in float32
                        orig_dtype = h.dtype
                        residual = (h.float() - h_after_first.float())  # (1, seq, hidden)

                        # Project residual onto SVD basis:
                        # coeffs[..., i] = residual @ Vh[i]
                        coeffs = torch.einsum('bsh,kh->bsk', residual, Vh_dev)
                        # coeffs shape: (1, seq, n_components)

                        if patch_mode == "topk":
                            # Zero out bottom directions
                            coeffs[:, :, patch_k:] = 0.0

                        elif patch_mode == "reverse_bottom":
                            # Keep top-k, reverse the rest
                            coeffs[:, :, patch_k:] = -coeffs[:, :, patch_k:]

                        # Reconstruct patched residual
                        patched_residual = torch.einsum('bsk,kh->bsh', coeffs, Vh_dev)

                        h = (h_after_first.float() + patched_residual).to(orig_dtype)

            # Final norm + LM head
            h = inner.norm(h)
            logits = model.lm_head(h)

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    generated = input_ids[0, prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# =============================================================================
# Step 5: Evaluation harness
# =============================================================================

def evaluate_variant(model, tokenizer, original_layers, layer_order,
                     block_start, block_end, first_end, second_end,
                     Vh, S, patch_mode, patch_k, label):
    """Run math_probe with a specific SVD patching variant."""

    def gen_fn(prompt):
        return generate_with_svd_patch(
            model, tokenizer, prompt, layer_order, original_layers,
            block_start, block_end, first_end, second_end,
            Vh, S, patch_mode, patch_k, max_new_tokens=64,
        )

    print(f"\n  Evaluating: {label}")
    result = run_math_probe(gen_fn, verbose=True)
    print(f"  --> {label}: math_score = {result['score']:.4f}")

    return {
        "label": label,
        "patch_mode": patch_mode,
        "patch_k": patch_k,
        "math_score": result["score"],
        "math_scores": result["scores"],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    t_global = time.time()
    print("=" * 70)
    print("SVD SUBSPACE PATCHING")
    print("Decompose the duplication residual; scale principal components independently.")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Calibration prompts: {len(CALIBRATION_PROMPTS)}")
    print(f"Top-k values: {TOP_K_VALUES}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # -- Load model --
    model, tokenizer = load_original_model(MODEL_PATH)
    device = next(model.parameters()).device
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)
    hidden_dim = model.config.hidden_size
    print(f"\nModel: {N} layers, hidden_dim={hidden_dim}, device={device}")

    all_results = {
        "model": MODEL_PATH,
        "num_layers": N,
        "hidden_dim": hidden_dim,
        "timestamp": datetime.now().isoformat(),
    }

    # -- Step 1: Find best block --
    best_start, best_end, disp_candidates = find_best_block_by_displacement(
        model, tokenizer, step=2
    )
    all_results["displacement_screening"] = {
        "best_block": [best_start, best_end],
        "candidates": disp_candidates,
    }

    block_start, block_end = best_start, best_end
    layer_order = build_layer_order(block_start, block_end, N)
    first_end, second_end = find_seam_positions(layer_order, block_start, block_end)

    # -- Step 2: Collect residuals --
    residuals, h1_list, h2_list = collect_residuals(
        model, tokenizer, block_start, block_end, CALIBRATION_PROMPTS
    )

    # -- Step 3: SVD --
    U, S, Vh = compute_svd(residuals)

    # Report SVD stats
    svd_stats = {
        "total_tokens": sum(r.shape[0] for r in residuals),
        "n_components": int(Vh.shape[0]),
        "top_singular_values": S[:32].tolist(),
        "variance_explained": {},
    }
    total_var = (S ** 2).sum().item()
    for k in TOP_K_VALUES:
        if k <= len(S):
            svd_stats["variance_explained"][str(k)] = float((S[:k] ** 2).sum().item() / total_var)
    all_results["svd_stats"] = svd_stats

    # Free calibration data
    del residuals, h1_list, h2_list, U
    gc.collect()
    torch.cuda.empty_cache()

    # -- Step 4+5: Evaluate variants --
    print(f"\n{'='*70}")
    print(f"STEP 4-5: Evaluating SVD-patched variants on block ({block_start},{block_end})")
    print(f"{'='*70}")

    eval_results = []

    # --- Baseline: no duplication (alpha=0) ---
    print(f"\n{'='*50}")
    print("BASELINE: alpha=0 (no duplication, just h1)")
    print(f"{'='*50}")
    r = evaluate_variant(
        model, tokenizer, original_layers, layer_order,
        block_start, block_end, first_end, second_end,
        Vh, S, "none", 0, "alpha=0 (no duplication)"
    )
    eval_results.append(r)

    # --- Baseline: full duplication (alpha=1) ---
    print(f"\n{'='*50}")
    print("BASELINE: alpha=1 (full duplication)")
    print(f"{'='*50}")
    r = evaluate_variant(
        model, tokenizer, original_layers, layer_order,
        block_start, block_end, first_end, second_end,
        Vh, S, "full", 0, "alpha=1 (full duplication)"
    )
    eval_results.append(r)

    # --- Top-k patching ---
    print(f"\n{'='*50}")
    print("TOP-K PATCHING: keep only top-k SVD directions")
    print(f"{'='*50}")
    for k in TOP_K_VALUES:
        if k > Vh.shape[0]:
            print(f"  Skipping k={k} (only {Vh.shape[0]} components available)")
            continue
        label = f"topk k={k}"
        r = evaluate_variant(
            model, tokenizer, original_layers, layer_order,
            block_start, block_end, first_end, second_end,
            Vh, S, "topk", k, label
        )
        eval_results.append(r)

    # --- Reverse bottom directions ---
    print(f"\n{'='*50}")
    print("REVERSE BOTTOM: keep top-k, negate bottom directions")
    print(f"{'='*50}")
    for k in TOP_K_VALUES:
        if k > Vh.shape[0]:
            print(f"  Skipping k={k} (only {Vh.shape[0]} components available)")
            continue
        label = f"reverse_bottom k={k}"
        r = evaluate_variant(
            model, tokenizer, original_layers, layer_order,
            block_start, block_end, first_end, second_end,
            Vh, S, "reverse_bottom", k, label
        )
        eval_results.append(r)

    all_results["evaluations"] = eval_results

    # -- Summary --
    elapsed = time.time() - t_global
    all_results["elapsed_minutes"] = elapsed / 60.0

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Block: ({block_start},{block_end})")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  SVD components: {Vh.shape[0]}")
    print()

    # Sort by score
    sorted_results = sorted(eval_results, key=lambda x: x["math_score"], reverse=True)
    print(f"  {'Rank':<6} {'Label':<35} {'Math Score':<12}")
    print(f"  {'-'*6} {'-'*35} {'-'*12}")
    for rank, r in enumerate(sorted_results, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  {rank:<6} {r['label']:<35} {r['math_score']:.4f}{marker}")

    # Find the best topk variant
    topk_results = [r for r in eval_results if r["patch_mode"] == "topk"]
    if topk_results:
        best_topk = max(topk_results, key=lambda x: x["math_score"])
        full_score = next((r["math_score"] for r in eval_results if r["patch_mode"] == "full"), 0)
        delta = best_topk["math_score"] - full_score
        print(f"\n  Best top-k: {best_topk['label']} (score={best_topk['math_score']:.4f}, "
              f"delta vs full: {delta:+.4f})")

    # Find the best reverse_bottom variant
    rev_results = [r for r in eval_results if r["patch_mode"] == "reverse_bottom"]
    if rev_results:
        best_rev = max(rev_results, key=lambda x: x["math_score"])
        full_score = next((r["math_score"] for r in eval_results if r["patch_mode"] == "full"), 0)
        delta = best_rev["math_score"] - full_score
        print(f"  Best reverse: {best_rev['label']} (score={best_rev['math_score']:.4f}, "
              f"delta vs full: {delta:+.4f})")

    print(f"\n  Total time: {elapsed/60:.1f} min")

    # -- Save --
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
