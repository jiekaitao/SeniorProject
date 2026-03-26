"""
BLOOD Impact Sweep — Does downstream smoothness change predict block quality?

Hypothesis: When a "good" block like (10,11) is duplicated, downstream layers become
SMOOTHER (lower BLOOD scores), because the second pass refines the hidden state into
something more in-distribution for downstream layers. Bad blocks make downstream layers
ROUGHER (higher BLOOD). If this holds, then "BLOOD impact" — the total downstream
smoothness change — should predict math probe delta better than displacement rho.

For each candidate config (i, j):
  1. Run base model on 4 diverse prompts, compute BLOOD at every layer.
  2. Apply duplication at (i, j), compute BLOOD at every layer.
  3. BLOOD impact = sum of (BLOOD_base - BLOOD_dup) at downstream layers.
     Positive = duplication made downstream layers smoother.
     Negative = duplication made them rougher.
  4. Correlate BLOOD impact with math_delta from brain scanner sweep.

We pick the top 10 + bottom 10 configs from the brain scanner sweep (by delta)
to get maximum spread for correlation analysis.

BLOOD computation uses Hutchinson's trace estimator:
  z ~ N(0, I),  Jz = autograd.grad(layer(h), h, z),  ||J||^2_F ~ ||Jz||^2
"""

import sys
import os
import json
import time
import gc
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, '/blue/cis4914/jietao/DeepPass/scripts')
from layer_duplicator import load_original_model

# =============================================================================
# Paths
# =============================================================================
MODEL_PATH = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"
SWEEP_PATH = "/blue/cis4914/jietao/DeepPass/results/sweep_7B/sweep_results.json"
SPECTRAL_PATH = "/blue/cis4914/jietao/DeepPass/results/spectral_7B_v2/spectral_results.json"
RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/blood_impact_sweep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 4 diverse prompts spanning different cognitive tasks
PROMPTS = [
    "What is 78313 multiplied by 88537?",
    "Explain how a transformer neural network processes text step by step.",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "Write a Python function to compute the Fibonacci sequence.",
]

N_HUTCHINSON = 5  # number of random vectors for trace estimation


# =============================================================================
# Config selection: top 10 + bottom 10 from brain scanner
# =============================================================================
def select_configs(sweep_path, n_top=10, n_bottom=10):
    """
    Load brain scanner sweep results and select the top-N and bottom-N configs
    by math probe delta. This gives maximum spread for correlation analysis.

    Returns:
        configs: list of (i, j, math_delta, math_score) tuples
        sweep_data: full sweep dict for later reference
    """
    with open(sweep_path) as f:
        sweep = json.load(f)

    items = []
    for key, val in sweep["results"].items():
        if "error" in val:
            continue
        i, j = map(int, key.split(","))
        items.append((i, j, val["delta"], val["score"]))

    # Sort by delta descending
    items.sort(key=lambda x: x[2], reverse=True)

    # Take top N and bottom N (avoid overlap if fewer configs)
    top = items[:n_top]
    bottom = items[-n_bottom:]

    # Merge, dedup by (i,j)
    seen = set()
    selected = []
    for item in top + bottom:
        key = (item[0], item[1])
        if key not in seen:
            seen.add(key)
            selected.append(item)

    return selected, sweep


# =============================================================================
# BLOOD computation (Jacobian Frobenius norm via Hutchinson estimator)
# =============================================================================
def compute_blood_all_layers(model, tokenizer, prompts, layer_order, n_hutchinson=5):
    """
    Compute BLOOD score (||J||^2_F estimate) at every layer in the given execution order.

    Uses the same approach as junction_confusion.py:
    - Forward pass with no_grad to collect hidden states at each layer boundary
    - For each layer, compute Jacobian-vector product via autograd.grad
    - Estimate ||J||^2_F ~ E[||Jz||^2] over n_hutchinson random vectors

    Args:
        model: loaded HF model (bfloat16)
        tokenizer: tokenizer
        prompts: list of prompt strings
        layer_order: list of layer indices defining execution order
        n_hutchinson: number of random vectors per layer

    Returns:
        blood_scores: dict mapping step_index -> mean BLOOD score across prompts
    """
    device = next(model.parameters()).device
    inner = model.model

    num_steps = len(layer_order)
    step_scores = defaultdict(list)  # step_idx -> list of scores (one per prompt)

    for prompt in prompts:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=128
        ).to(device)
        input_ids = inputs["input_ids"]

        # Phase 1: forward pass to collect all boundary hidden states (no grad)
        with torch.no_grad():
            h_embed = inner.embed_tokens(input_ids)
            seq_len = h_embed.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h_embed, pos_ids)

        boundary_h = [h_embed.detach()]
        h = h_embed
        with torch.no_grad():
            for step_idx, layer_idx in enumerate(layer_order):
                out = inner.layers[layer_idx](
                    h, position_embeddings=pos_embeds, use_cache=False
                )
                h = out[0] if isinstance(out, tuple) else out
                boundary_h.append(h.detach())

        # Phase 2: compute Jacobian norm at each layer (needs grad for one layer)
        for step_idx, layer_idx in enumerate(layer_order):
            h_in = boundary_h[step_idx].detach().clone()

            jac_norm_estimates = []
            for _ in range(n_hutchinson):
                z = torch.randn_like(h_in)
                h_in_grad = h_in.detach().clone().requires_grad_(True)

                out = inner.layers[layer_idx](
                    h_in_grad, position_embeddings=pos_embeds, use_cache=False
                )
                h_out = out[0] if isinstance(out, tuple) else out

                # Compute Jz via autograd (J = d(h_out)/d(h_in))
                Jz = torch.autograd.grad(
                    h_out, h_in_grad, grad_outputs=z, create_graph=False
                )[0]
                jac_norm_sq = (Jz ** 2).sum().item()
                jac_norm_estimates.append(jac_norm_sq)

            blood_score = float(np.mean(jac_norm_estimates))
            step_scores[step_idx].append(blood_score)

        # Free memory
        del boundary_h, h_embed, h
        torch.cuda.empty_cache()

    # Average across prompts
    blood_means = {}
    for step_idx in range(num_steps):
        if step_idx in step_scores:
            blood_means[step_idx] = float(np.mean(step_scores[step_idx]))
        else:
            blood_means[step_idx] = 0.0

    return blood_means


# =============================================================================
# BLOOD impact computation for a single config
# =============================================================================
def compute_blood_impact(model, tokenizer, num_layers, i, j,
                         base_blood, prompts, n_hutchinson=5):
    """
    Compute the BLOOD impact of duplicating block [i, j).

    BLOOD impact = sum over downstream layers of (BLOOD_base - BLOOD_dup).
    Positive = duplication made downstream layers smoother (good).
    Negative = duplication made them rougher (bad).

    "Downstream" = all layers after the duplicated block in the base model,
    i.e., layers j, j+1, ..., N-1.

    In the duplicated execution order:
      [0, ..., j-1, i, ..., j-1, j, ..., N-1]
    The base layers j..N-1 appear at positions (j + (j-i)) .. (j + (j-i) + N-1-j).

    We compare BLOOD at those positions against BLOOD at positions j..N-1 in base.
    """
    block_size = j - i

    # Build duplicated layer order
    dup_order = list(range(j)) + list(range(i, j)) + list(range(j, num_layers))

    # Compute BLOOD on duplicated model
    dup_blood = compute_blood_all_layers(
        model, tokenizer, prompts, dup_order, n_hutchinson=n_hutchinson
    )

    # Map downstream layers: base step k (for k >= j) maps to dup step k + block_size
    downstream_deltas = []
    per_layer_detail = {}
    for base_step in range(j, num_layers):
        dup_step = base_step + block_size
        b_blood = base_blood.get(base_step, 0.0)
        d_blood = dup_blood.get(dup_step, 0.0)
        delta = b_blood - d_blood  # positive = smoother after duplication
        downstream_deltas.append(delta)
        per_layer_detail[base_step] = {
            "base_blood": b_blood,
            "dup_blood": d_blood,
            "delta": delta,
        }

    blood_impact = sum(downstream_deltas)
    mean_impact = float(np.mean(downstream_deltas)) if downstream_deltas else 0.0

    # Also collect BLOOD at the junction point and duplicated layers
    junction_step = j + block_size  # first layer after duplicated block in dup order
    junction_blood = dup_blood.get(junction_step, None)
    base_junction_blood = base_blood.get(j, None)

    return {
        "blood_impact": blood_impact,
        "mean_downstream_delta": mean_impact,
        "num_downstream_layers": len(downstream_deltas),
        "junction_blood_base": base_junction_blood,
        "junction_blood_dup": junction_blood,
        "per_layer_detail": per_layer_detail,
        "dup_blood_all": dup_blood,
    }


# =============================================================================
# Correlation analysis
# =============================================================================
def analyze_correlations(results, sweep_data, spectral_path=None):
    """
    Correlate BLOOD impact with math probe delta. Optionally compare against
    displacement rho if spectral results are available.
    """
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS: What predicts block quality?")
    print("=" * 70)

    # Extract arrays from BLOOD sweep
    configs = []
    math_deltas = []
    blood_impacts = []

    for entry in results:
        configs.append(f"{entry['i']},{entry['j']}")
        math_deltas.append(entry["math_delta"])
        blood_impacts.append(entry["blood_impact"])

    math_deltas = np.array(math_deltas)
    blood_impacts = np.array(blood_impacts)

    # 1. BLOOD impact vs math delta
    r_blood, p_blood = spearmanr(blood_impacts, math_deltas)
    r_blood_pearson, p_blood_pearson = pearsonr(blood_impacts, math_deltas)

    print(f"\n  BLOOD impact vs math_delta:")
    print(f"    Spearman r = {r_blood:+.4f}  (p = {p_blood:.6f})")
    print(f"    Pearson  r = {r_blood_pearson:+.4f}  (p = {p_blood_pearson:.6f})")
    sig = "***" if p_blood < 0.001 else "**" if p_blood < 0.01 else "*" if p_blood < 0.05 else "ns"
    print(f"    Significance: {sig}")

    correlation_results = {
        "blood_impact_vs_math_delta": {
            "spearman_r": float(r_blood),
            "spearman_p": float(p_blood),
            "pearson_r": float(r_blood_pearson),
            "pearson_p": float(p_blood_pearson),
        }
    }

    # 2. If spectral results available, compare with displacement rho
    if spectral_path and os.path.exists(spectral_path):
        with open(spectral_path) as f:
            spectral = json.load(f)

        disp_rhos = []
        blood_for_common = []
        math_for_common = []
        common_configs = []

        for entry in results:
            key = f"{entry['i']},{entry['j']}"
            if key in spectral["results"]:
                s = spectral["results"][key]
                if "error" not in s and "displacement_rho" in s:
                    disp_rhos.append(s["displacement_rho"])
                    blood_for_common.append(entry["blood_impact"])
                    math_for_common.append(entry["math_delta"])
                    common_configs.append(key)

        if len(disp_rhos) >= 5:
            disp_rhos = np.array(disp_rhos)
            blood_for_common = np.array(blood_for_common)
            math_for_common = np.array(math_for_common)

            # Displacement rho vs math delta (negative correlation expected:
            # lower displacement = better)
            r_disp, p_disp = spearmanr(disp_rhos, math_for_common)
            r_disp_pearson, _ = pearsonr(disp_rhos, math_for_common)

            # BLOOD impact on same subset
            r_blood_sub, p_blood_sub = spearmanr(blood_for_common, math_for_common)

            print(f"\n  Displacement rho vs math_delta (n={len(disp_rhos)} overlapping configs):")
            print(f"    Spearman r = {r_disp:+.4f}  (p = {p_disp:.6f})")
            print(f"    Pearson  r = {r_disp_pearson:+.4f}")
            sig_d = "***" if p_disp < 0.001 else "**" if p_disp < 0.01 else "*" if p_disp < 0.05 else "ns"
            print(f"    Significance: {sig_d}")

            print(f"\n  BLOOD impact vs math_delta (same {len(disp_rhos)} configs):")
            print(f"    Spearman r = {r_blood_sub:+.4f}  (p = {p_blood_sub:.6f})")

            correlation_results["displacement_rho_vs_math_delta"] = {
                "spearman_r": float(r_disp),
                "spearman_p": float(p_disp),
                "pearson_r": float(r_disp_pearson),
                "n_configs": len(disp_rhos),
            }

            # 3. Combined predictor: z-score normalize and sum
            blood_z = (blood_for_common - blood_for_common.mean()) / (blood_for_common.std() + 1e-10)
            # Negate displacement rho since lower is better
            disp_z = -(disp_rhos - disp_rhos.mean()) / (disp_rhos.std() + 1e-10)
            combined = blood_z + disp_z

            r_comb, p_comb = spearmanr(combined, math_for_common)
            print(f"\n  Combined (BLOOD_z - disp_z) vs math_delta:")
            print(f"    Spearman r = {r_comb:+.4f}  (p = {p_comb:.6f})")

            correlation_results["combined_vs_math_delta"] = {
                "spearman_r": float(r_comb),
                "spearman_p": float(p_comb),
            }

            # Winner
            print(f"\n  {'=' * 50}")
            abs_blood = abs(r_blood_sub)
            abs_disp = abs(r_disp)
            abs_comb = abs(r_comb)
            if abs_comb >= abs_blood and abs_comb >= abs_disp:
                print(f"  WINNER: Combined predictor (|r| = {abs_comb:.4f})")
            elif abs_blood >= abs_disp:
                print(f"  WINNER: BLOOD impact (|r| = {abs_blood:.4f} vs displacement |r| = {abs_disp:.4f})")
            else:
                print(f"  WINNER: Displacement rho (|r| = {abs_disp:.4f} vs BLOOD |r| = {abs_blood:.4f})")
            print(f"  {'=' * 50}")
        else:
            print(f"\n  Only {len(disp_rhos)} overlapping spectral configs found, skipping comparison.")
    else:
        print("\n  No spectral results found, skipping displacement rho comparison.")

    return correlation_results


# =============================================================================
# Plotting
# =============================================================================
def plot_results(results, correlation_results, output_dir):
    """Generate scatter plots of BLOOD impact vs math delta."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    output_dir = Path(output_dir)

    math_deltas = np.array([r["math_delta"] for r in results])
    blood_impacts = np.array([r["blood_impact"] for r in results])
    configs = [f"({r['i']},{r['j']})" for r in results]

    # --- Plot 1: BLOOD impact vs math delta ---
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['green' if d > 0 else 'red' for d in math_deltas]
    ax.scatter(blood_impacts, math_deltas, c=colors, s=60, alpha=0.7, edgecolors='black', linewidths=0.5)

    # Annotate each point with config
    for idx, cfg in enumerate(configs):
        ax.annotate(cfg, (blood_impacts[idx], math_deltas[idx]),
                    fontsize=6, alpha=0.7, ha='center', va='bottom')

    # Trend line
    if len(blood_impacts) > 2:
        z = np.polyfit(blood_impacts, math_deltas, 1)
        p = np.poly1d(z)
        x_range = np.linspace(blood_impacts.min(), blood_impacts.max(), 100)
        ax.plot(x_range, p(x_range), '--', color='gray', alpha=0.5, linewidth=1)

    r_val = correlation_results.get("blood_impact_vs_math_delta", {}).get("spearman_r", 0)
    p_val = correlation_results.get("blood_impact_vs_math_delta", {}).get("spearman_p", 1)

    ax.set_xlabel("BLOOD Impact (positive = smoother downstream)", fontsize=12)
    ax.set_ylabel("Math Probe Delta", fontsize=12)
    ax.set_title(f"BLOOD Impact vs Math Probe Delta\n"
                 f"Spearman r = {r_val:+.4f} (p = {p_val:.4f})", fontsize=13)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.4)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / "blood_impact_vs_math_delta.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'blood_impact_vs_math_delta.png'}")

    # --- Plot 2: Per-config BLOOD profile (base vs dup) for top and bottom configs ---
    # Show the top 3 and bottom 3 configs side by side
    sorted_results = sorted(results, key=lambda r: r["math_delta"], reverse=True)
    showcase = sorted_results[:3] + sorted_results[-3:]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax_idx, entry in enumerate(showcase):
        row = ax_idx // 3
        col = ax_idx % 3
        ax = axes[row, col]

        detail = entry.get("per_layer_detail", {})
        if not detail:
            continue

        layers_sorted = sorted(detail.keys(), key=lambda k: int(k))
        base_vals = [detail[k]["base_blood"] for k in layers_sorted]
        dup_vals = [detail[k]["dup_blood"] for k in layers_sorted]
        layer_nums = [int(k) for k in layers_sorted]

        ax.plot(layer_nums, base_vals, 'b-o', markersize=3, label='Base', alpha=0.7)
        ax.plot(layer_nums, dup_vals, 'r-s', markersize=3, label='Dup', alpha=0.7)
        ax.fill_between(layer_nums, base_vals, dup_vals, alpha=0.15,
                        color='green' if entry["blood_impact"] > 0 else 'red')

        title_prefix = "GOOD" if entry["math_delta"] > 0 else "BAD"
        ax.set_title(f"{title_prefix}: ({entry['i']},{entry['j']}) "
                     f"math_d={entry['math_delta']:+.3f}\n"
                     f"BLOOD impact={entry['blood_impact']:+.1f}",
                     fontsize=9)
        ax.set_xlabel("Layer", fontsize=8)
        ax.set_ylabel("BLOOD (||J||^2_F)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    plt.suptitle("Downstream BLOOD Profiles: Good Configs (top) vs Bad Configs (bottom)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "blood_profiles_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'blood_profiles_comparison.png'}")


# =============================================================================
# Main
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="BLOOD Impact Sweep — measure downstream smoothness change "
                    "from layer duplication and correlate with math probe quality."
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH,
        help="Path to HuggingFace model"
    )
    parser.add_argument(
        "--sweep", type=str, default=SWEEP_PATH,
        help="Path to brain scanner sweep_results.json"
    )
    parser.add_argument(
        "--spectral", type=str, default=SPECTRAL_PATH,
        help="Path to spectral_results.json (for displacement rho comparison)"
    )
    parser.add_argument(
        "--output", type=str, default=str(RESULTS_DIR),
        help="Output directory"
    )
    parser.add_argument(
        "--n-top", type=int, default=10,
        help="Number of top configs from brain scanner to include"
    )
    parser.add_argument(
        "--n-bottom", type=int, default=10,
        help="Number of bottom configs from brain scanner to include"
    )
    parser.add_argument(
        "--hutchinson-samples", type=int, default=N_HUTCHINSON,
        help="Number of Hutchinson random vectors per layer (default: 5)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # =========================================================================
    # Step 1: Select configs from brain scanner
    # =========================================================================
    print("=" * 70)
    print("BLOOD IMPACT SWEEP")
    print("=" * 70)

    selected, sweep_data = select_configs(args.sweep, args.n_top, args.n_bottom)
    print(f"\nSelected {len(selected)} configs from brain scanner sweep:")
    print(f"  Top {args.n_top} by math delta + Bottom {args.n_bottom} by math delta")
    print()
    for i, j, delta, score in selected:
        tag = "TOP" if delta > 0 else "BOT"
        print(f"  [{tag}] ({i:2d},{j:2d})  math_delta={delta:+.4f}  score={score:.4f}")

    # =========================================================================
    # Step 2: Load model
    # =========================================================================
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_original_model(args.model)
    model.eval()

    inner = model.model
    num_layers = len(inner.layers)
    print(f"Model has {num_layers} layers")

    # =========================================================================
    # Step 3: Compute base BLOOD scores (once, shared across all configs)
    # =========================================================================
    print(f"\n--- Computing BASE BLOOD scores ({num_layers} layers, "
          f"{len(PROMPTS)} prompts, {args.hutchinson_samples} Hutchinson samples) ---")
    base_order = list(range(num_layers))
    t0 = time.time()
    base_blood = compute_blood_all_layers(
        model, tokenizer, PROMPTS, base_order, n_hutchinson=args.hutchinson_samples
    )
    t_base = time.time() - t0
    print(f"  Base BLOOD computed in {t_base:.1f}s")

    # Print base BLOOD profile
    print(f"\n  Base BLOOD profile:")
    for step in range(num_layers):
        val = base_blood.get(step, 0)
        bar = "#" * min(int(val / 1e6), 50)
        print(f"    Layer {step:2d}: {val:12.1f}  {bar}")

    # =========================================================================
    # Step 4: Compute BLOOD impact for each selected config
    # =========================================================================
    all_results = []
    for config_idx, (i, j, math_delta, math_score) in enumerate(selected):
        print(f"\n--- Config {config_idx + 1}/{len(selected)}: ({i},{j}) "
              f"math_delta={math_delta:+.4f} ---")
        t0 = time.time()

        impact_data = compute_blood_impact(
            model, tokenizer, num_layers, i, j,
            base_blood, PROMPTS, n_hutchinson=args.hutchinson_samples
        )

        elapsed = time.time() - t0
        print(f"  BLOOD impact = {impact_data['blood_impact']:+.1f} "
              f"(mean downstream delta = {impact_data['mean_downstream_delta']:+.1f})")
        print(f"  Junction BLOOD: base={impact_data['junction_blood_base']:.1f} "
              f"-> dup={impact_data['junction_blood_dup']:.1f}" if impact_data['junction_blood_dup'] else "")
        print(f"  Computed in {elapsed:.1f}s")

        entry = {
            "i": i,
            "j": j,
            "config": f"{i},{j}",
            "math_delta": math_delta,
            "math_score": math_score,
            "blood_impact": impact_data["blood_impact"],
            "mean_downstream_delta": impact_data["mean_downstream_delta"],
            "num_downstream_layers": impact_data["num_downstream_layers"],
            "junction_blood_base": impact_data["junction_blood_base"],
            "junction_blood_dup": impact_data["junction_blood_dup"],
            "per_layer_detail": {
                str(k): v for k, v in impact_data["per_layer_detail"].items()
            },
        }
        all_results.append(entry)

        # Save intermediate results after each config
        intermediate = {
            "status": "in_progress",
            "completed": config_idx + 1,
            "total": len(selected),
            "base_blood": {str(k): v for k, v in base_blood.items()},
            "results": all_results,
        }
        with open(output_dir / "blood_impact_results.json", "w") as f:
            json.dump(intermediate, f, indent=2)

        # Free VRAM between configs
        torch.cuda.empty_cache()

    # =========================================================================
    # Step 5: Correlation analysis
    # =========================================================================
    correlation_results = analyze_correlations(all_results, sweep_data, args.spectral)

    # =========================================================================
    # Step 6: Print summary table
    # =========================================================================
    print(f"\n{'=' * 90}")
    print(f"{'Config':>10} {'math_delta':>12} {'BLOOD_impact':>14} {'mean_dwnstrm':>14} "
          f"{'jnct_base':>12} {'jnct_dup':>12}")
    print(f"{'-' * 90}")

    sorted_by_blood = sorted(all_results, key=lambda r: r["blood_impact"], reverse=True)
    for entry in sorted_by_blood:
        jb = entry.get("junction_blood_base")
        jd = entry.get("junction_blood_dup")
        print(f"  ({entry['i']:2d},{entry['j']:2d}) {entry['math_delta']:+12.4f} "
              f"{entry['blood_impact']:+14.1f} {entry['mean_downstream_delta']:+14.1f} "
              f"{jb:12.1f} {(jd if jd else 0):12.1f}")

    # =========================================================================
    # Step 7: Generate plots
    # =========================================================================
    print(f"\n--- Generating Plots ---")
    plot_results(all_results, correlation_results, output_dir)

    # =========================================================================
    # Step 8: Save final results
    # =========================================================================
    total_time = time.time() - t_start

    final_output = {
        "status": "complete",
        "model_path": args.model,
        "num_configs": len(all_results),
        "prompts": PROMPTS,
        "n_hutchinson": args.hutchinson_samples,
        "total_time_seconds": total_time,
        "base_blood": {str(k): v for k, v in base_blood.items()},
        "results": all_results,
        "correlations": correlation_results,
    }
    with open(output_dir / "blood_impact_results.json", "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nResults saved to {output_dir / 'blood_impact_results.json'}")

    print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Results directory: {output_dir}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
