"""
DeepPass Unified Analysis

Combines TRM's theoretical framework with RYS empirical methodology.

Key insight from TRM research:
- Perturbation rho >> 1 but displacement rho < 1
- This means the map is expansive in random directions but contractive
  along the actual solution trajectory
- A block is a good RYS candidate when displacement rho ∈ [0.5, 0.95]
  AND the residual ||F(F(h)) - F(h)|| is significant

This script computes the "DeepPass Score" for each layer block:
1. Displacement rho (convergence along trajectory)
2. Residual magnitude (room for improvement from second pass)
3. Perturbation rho (representation sharpening)
4. Layer position (avoid encoding/decoding layers)

Then validates against actual brain scanner results (if available).
"""

import json, sys, os, time, torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_deeppass_score(
    model, tokenizer, layers, start, end, N,
    test_inputs, inner_model, eps=1e-3
):
    """
    Compute DeepPass Score for block [start, end).

    Score = f(displacement_rho, residual, position)

    Higher score = better candidate for duplication.
    """
    device = next(model.parameters()).device

    disp_ratios = []
    residuals = []
    pert_ratios = []

    for input_ids, attention_mask in test_inputs:
        with torch.no_grad():
            # Get embeddings
            if hasattr(inner_model, 'embed_tokens'):
                h = inner_model.embed_tokens(input_ids)
            elif hasattr(inner_model, 'wte'):
                h = inner_model.wte(input_ids)
            else:
                continue

            pos_ids = torch.arange(h.shape[1], device=device).unsqueeze(0)

            # Run layers [0, start)
            for idx in range(start):
                out = layers[idx](h, attention_mask=attention_mask, position_ids=pos_ids)
                h = out[0] if isinstance(out, tuple) else out

            h_input = h  # Input to the block

            # First pass: F(h)
            h1 = h_input.clone()
            for idx in range(start, end):
                out = layers[idx](h1, attention_mask=attention_mask, position_ids=pos_ids)
                h1 = out[0] if isinstance(out, tuple) else out

            # Second pass: F(F(h))
            h2 = h1.clone()
            for idx in range(start, end):
                out = layers[idx](h2, attention_mask=attention_mask, position_ids=pos_ids)
                h2 = out[0] if isinstance(out, tuple) else out

            # Perturbation: F(h + eps)
            perturbation = torch.randn_like(h_input) * eps
            h_pert = h_input + perturbation
            for idx in range(start, end):
                out = layers[idx](h_pert, attention_mask=attention_mask, position_ids=pos_ids)
                h_pert = out[0] if isinstance(out, tuple) else out

            # Compute metrics (per-token, then average)
            diff1 = (h1 - h_input).float().norm(dim=-1).mean().item()
            diff2 = (h2 - h1).float().norm(dim=-1).mean().item()
            pert_out = (h_pert - h1).float().norm(dim=-1).mean().item()
            pert_in = perturbation.float().norm(dim=-1).mean().item()

            if diff1 > 1e-10:
                disp_ratios.append(diff2 / diff1)
            residuals.append(diff2)
            if pert_in > 1e-10:
                pert_ratios.append(pert_out / pert_in)

    disp_rho = np.mean(disp_ratios) if disp_ratios else 1.0
    residual = np.mean(residuals) if residuals else 0.0
    pert_rho = np.mean(pert_ratios) if pert_ratios else 1.0

    # DeepPass Score computation
    # Based on TRM research findings:
    # Best candidates have disp_rho in [0.5, 0.95] with high residual

    # Contraction component: peaks at rho ≈ 0.7, drops at 0 and 1
    if disp_rho < 1.0:
        contraction = (1 - disp_rho) * min(disp_rho / 0.7, 1.0)
    else:
        contraction = max(0, 1 - (disp_rho - 1) * 5)  # Penalty for expansive

    # Residual component: higher = more room for improvement
    # Normalize by expecting typical values in [0.1, 10]
    residual_score = np.tanh(residual)

    # Position: middle layers are better (avoid first/last 15%)
    center = (start + end) / 2
    edge_dist = min(center, N - center)
    position_score = min(1.0, edge_dist / (N * 0.15))

    # Block size preference: 5-10 layers optimal (from Ng's findings)
    bs = end - start
    size_score = np.exp(-((bs - 7) / 5) ** 2)

    # Combined DeepPass Score
    score = (contraction * 0.35 +
             residual_score * 0.30 +
             position_score * 0.20 +
             size_score * 0.15)

    return {
        "displacement_rho": float(disp_rho),
        "residual": float(residual),
        "perturbation_rho": float(pert_rho),
        "contraction_score": float(contraction),
        "residual_score": float(residual_score),
        "position_score": float(position_score),
        "size_score": float(size_score),
        "deeppass_score": float(score),
    }


def run_deeppass_analysis(
    model_path: str,
    output_dir: str,
    block_sizes: list = None,
    step: int = 1,
    num_test_prompts: int = 8,
    top_k_eval: int = 10,
):
    """
    Full DeepPass analysis pipeline:
    1. Compute DeepPass Score for all blocks (cheap)
    2. Rank blocks by score
    3. Run actual math probe evaluation on top-K (expensive, but targeted)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.eval()

    inner = model.model if hasattr(model, 'model') else model.transformer
    layers = list(inner.layers if hasattr(inner, 'layers') else inner.h)
    attr_name = 'layers' if hasattr(inner, 'layers') else 'h'
    N = len(layers)
    print(f"Model has {N} layers")

    if block_sizes is None:
        block_sizes = list(range(3, min(N // 3, 15) + 1))

    # Prepare test inputs
    prompts = [
        "The theory of general relativity",
        "In Python, a list comprehension",
        "The square root of 144 is",
        "Machine learning algorithms learn",
        "The chemical formula for water",
        "To prove a theorem by induction",
        "The GDP of the United States",
        "A recursive function calls itself",
    ][:num_test_prompts]

    test_inputs = []
    for p in prompts:
        tok = tokenizer(p, return_tensors="pt", padding=True,
                       truncation=True, max_length=64).to(model.device)
        test_inputs.append((tok['input_ids'], tok.get('attention_mask')))

    # Phase 1: Compute DeepPass Scores for all blocks
    print(f"\n{'='*60}")
    print("PHASE 1: Computing DeepPass Scores")
    print(f"{'='*60}")

    all_scores = {}
    configs = []
    for bs in block_sizes:
        for start in range(0, N - bs + 1, step):
            configs.append((start, start + bs))

    t0 = time.time()
    for idx, (start, end) in enumerate(configs):
        try:
            metrics = compute_deeppass_score(
                model, tokenizer, layers, start, end, N,
                test_inputs, inner
            )
            all_scores[(start, end)] = metrics
            bs = end - start
            print(f"  [{idx+1:4d}/{len(configs)}] ({start:3d},{end:3d}) bs={bs:2d} "
                  f"disp_ρ={metrics['displacement_rho']:.4f} "
                  f"resid={metrics['residual']:.4f} "
                  f"score={metrics['deeppass_score']:.4f}")
        except Exception as e:
            print(f"  [{idx+1:4d}/{len(configs)}] ({start:3d},{end:3d}) ERROR: {e}")

    phase1_time = time.time() - t0
    print(f"\nPhase 1 complete in {phase1_time:.1f}s ({phase1_time/60:.1f}m)")

    # Save phase 1 results
    save_data = {
        "num_layers": N,
        "model_path": model_path,
        "phase1_time_seconds": phase1_time,
        "scores": {f"{s},{e}": v for (s, e), v in all_scores.items()},
    }
    with open(output_dir / "deeppass_scores.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Rank by DeepPass Score
    ranked = sorted(all_scores.items(), key=lambda x: x[1]["deeppass_score"], reverse=True)

    print(f"\n{'='*60}")
    print(f"TOP {min(top_k_eval, len(ranked))} CANDIDATES BY DEEPPASS SCORE")
    print(f"{'='*60}")
    print(f"{'Rank':>4} {'Config':>10} {'Score':>8} {'Disp ρ':>8} {'Resid':>8} {'Pert ρ':>8}")
    for i, ((s, e), m) in enumerate(ranked[:top_k_eval], 1):
        print(f"{i:4d} ({s:3d},{e:3d}) {m['deeppass_score']:8.4f} "
              f"{m['displacement_rho']:8.4f} {m['residual']:8.4f} "
              f"{m['perturbation_rho']:8.4f}")

    # Phase 2: Validate top-K with actual math probe
    print(f"\n{'='*60}")
    print(f"PHASE 2: Evaluating top {top_k_eval} candidates with math probe")
    print(f"{'='*60}")

    # Also evaluate baseline
    print("\n--- Baseline ---")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    baseline = run_math_probe(gen_fn, verbose=False)
    baseline_score = baseline['score']
    print(f"  Baseline math score: {baseline_score:.4f}")

    eval_results = {}
    for rank, ((start, end), metrics) in enumerate(ranked[:top_k_eval], 1):
        bs = end - start
        print(f"\n--- Rank {rank}: ({start},{end}) bs={bs} deeppass_score={metrics['deeppass_score']:.4f} ---")

        # Apply duplication
        layer_order = list(range(end)) + list(range(start, end)) + list(range(end, N))
        new_layers = [layers[k] for k in layer_order]
        setattr(inner, attr_name, nn.ModuleList(new_layers))
        model.config.num_hidden_layers = len(new_layers)

        result = run_math_probe(gen_fn, verbose=False)
        delta = result['score'] - baseline_score

        eval_results[(start, end)] = {
            "math_score": result['score'],
            "math_delta": delta,
            "deeppass_score": metrics['deeppass_score'],
            "rank": rank,
        }
        print(f"  Math score: {result['score']:.4f} (delta: {delta:+.4f})")

        # Restore
        setattr(inner, attr_name, nn.ModuleList(layers))
        model.config.num_hidden_layers = N

    # Summary
    print(f"\n{'='*60}")
    print("DEEPPASS ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Layers: {N}")
    print(f"Phase 1 (scoring): {phase1_time:.1f}s for {len(configs)} configs")
    print(f"Phase 2 (eval): {top_k_eval} configs evaluated")
    print(f"\nBaseline math: {baseline_score:.4f}")
    print(f"\n{'Rank':>4} {'Config':>10} {'DP Score':>10} {'Math Δ':>10} {'Match':>6}")

    hits = 0
    for rank, ((s, e), _) in enumerate(ranked[:top_k_eval], 1):
        if (s, e) in eval_results:
            er = eval_results[(s, e)]
            match = "YES" if er['math_delta'] > 0 else "no"
            if er['math_delta'] > 0:
                hits += 1
            print(f"{rank:4d} ({s:3d},{e:3d}) {er['deeppass_score']:10.4f} "
                  f"{er['math_delta']:+10.4f} {match:>6}")

    hit_rate = hits / min(top_k_eval, len(eval_results)) if eval_results else 0
    print(f"\nPrediction accuracy: {hits}/{min(top_k_eval, len(eval_results))} = {hit_rate:.0%}")
    print(f"(Rate of top-ranked DeepPass configs that actually improve math score)")

    # Save everything
    save_data["phase2_results"] = {f"{s},{e}": v for (s, e), v in eval_results.items()}
    save_data["baseline_math_score"] = baseline_score
    save_data["prediction_hit_rate"] = hit_rate
    with open(output_dir / "deeppass_full_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    return all_scores, eval_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--block-sizes", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.output is None:
        name = Path(args.model).name
        args.output = f"/blue/cis4914/jietao/DeepPass/results/deeppass_{name}"

    block_sizes = None
    if args.block_sizes:
        block_sizes = [int(x) for x in args.block_sizes.split(",")]

    run_deeppass_analysis(args.model, args.output, block_sizes=block_sizes,
                          step=args.step, top_k_eval=args.top_k)
