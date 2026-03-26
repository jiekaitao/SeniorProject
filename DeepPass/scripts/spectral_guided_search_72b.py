"""
DeepPass Spectral-Guided Search on 72B

Instead of Ng's brute-force 3,241-config sweep, use spectral analysis to
identify the top candidate blocks, then validate only those with the math probe.

Goal: Find a better (i,j) config than Ng's (45,52) using 10x fewer evaluations.
"""

import sys, os, json, time, torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe


def spectral_screen_72b(model, tokenizer, layers, N, inner, attr_name, prompts):
    """Compute displacement metrics for candidate blocks on 72B."""
    device = next(model.parameters()).device

    # Focus on middle layers (20-65) with block sizes 5-10
    # This is where Ng found the optimal config
    candidates = []
    for bs in [5, 6, 7, 8, 9, 10]:
        for start in range(20, 66 - bs, 2):  # step=2 to save time
            candidates.append((start, start + bs))

    print(f"Screening {len(candidates)} candidate blocks...")

    scored = []
    original_layers = list(layers)

    for idx, (start, end) in enumerate(candidates):
        try:
            # Run model normally
            tok_inputs = tokenizer(prompts[0], return_tensors="pt", truncation=True,
                                   max_length=64).to(device)

            with torch.no_grad():
                out_normal = model(**tok_inputs, use_cache=False)
                logits_normal = out_normal.logits.detach()

                # Run with duplication
                layer_order = list(range(end)) + list(range(start, end)) + list(range(end, N))
                dup_layers = [original_layers[k] for k in layer_order]
                setattr(inner, attr_name, nn.ModuleList(dup_layers))
                model.config.num_hidden_layers = len(dup_layers)

                out_dup = model(**tok_inputs, use_cache=False)
                logits_dup = out_dup.logits.detach()

                # Restore
                setattr(inner, attr_name, nn.ModuleList(original_layers))
                model.config.num_hidden_layers = N

                # Displacement metric
                diff = (logits_dup - logits_normal).float()
                disp = diff.norm(dim=-1).mean().item()
                norm = logits_normal.float().norm(dim=-1).mean().item()
                disp_ratio = disp / norm if norm > 1e-10 else 0

                scored.append({
                    "config": (start, end),
                    "displacement": disp,
                    "displacement_ratio": disp_ratio,
                    "block_size": end - start,
                })

                if (idx + 1) % 10 == 0:
                    print(f"  [{idx+1}/{len(candidates)}] ({start},{end}) disp_ratio={disp_ratio:.4f}")

        except Exception as e:
            print(f"  [{idx+1}/{len(candidates)}] ({start},{end}) ERROR: {e}")
            setattr(inner, attr_name, nn.ModuleList(original_layers))
            model.config.num_hidden_layers = N

    return scored


def main():
    MODEL_PATH = "/blue/cis4914/jietao/DeepPass/models/full/calme-2.1-qwen2-72b"
    RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/spectral_guided_72b")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prompts = [
        "What is 123456789 multiplied by 987654321? The answer is",
        "The theory of general relativity describes",
        "def fibonacci(n):\n    if n <= 1:",
        "The square root of 152399025 is",
    ]

    print("Loading 72B model...")
    model, tokenizer = load_original_model(MODEL_PATH)

    inner = model.model
    attr_name = 'layers'
    layers = list(inner.layers)
    N = len(layers)
    original_layers = list(layers)
    print(f"Model has {N} layers")

    # Phase 1: Spectral screening
    print("\n" + "=" * 60)
    print("PHASE 1: Spectral Screening")
    print("=" * 60)

    scored = spectral_screen_72b(model, tokenizer, layers, N, inner, attr_name, prompts)

    # Sort by displacement ratio (moderate displacement = most interesting)
    # Too low = block does nothing, too high = block is destabilizing
    for s in scored:
        # Score: prefer moderate displacement in middle layers
        center = (s["config"][0] + s["config"][1]) / 2
        edge_dist = min(center - 20, 65 - center)
        s["search_score"] = s["displacement_ratio"] * min(1.0, edge_dist / 10)

    scored.sort(key=lambda x: x["search_score"], reverse=True)

    print(f"\nTop 20 candidates by spectral score:")
    for i, s in enumerate(scored[:20]):
        print(f"  {i+1:2d}. ({s['config'][0]:3d},{s['config'][1]:3d}) "
              f"disp_ratio={s['displacement_ratio']:.4f} score={s['search_score']:.4f}")

    # Phase 2: Evaluate top candidates with math probe
    print("\n" + "=" * 60)
    print("PHASE 2: Math Probe Evaluation (top 15 + Ng's baseline)")
    print("=" * 60)

    # Baseline
    print("\n--- Baseline ---")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    baseline = run_math_probe(gen_fn, verbose=False)
    print(f"  Baseline: {baseline['score']:.4f}")

    # Ng's config for comparison
    print("\n--- Ng's config (45, 52) ---")
    layer_order = list(range(52)) + list(range(45, 52)) + list(range(52, N))
    dup_layers = [original_layers[k] for k in layer_order]
    setattr(inner, attr_name, nn.ModuleList(dup_layers))
    model.config.num_hidden_layers = len(dup_layers)
    ng_result = run_math_probe(gen_fn, verbose=False)
    ng_delta = ng_result['score'] - baseline['score']
    print(f"  Ng (45,52): {ng_result['score']:.4f} (delta: {ng_delta:+.4f})")
    setattr(inner, attr_name, nn.ModuleList(original_layers))
    model.config.num_hidden_layers = N

    # Test top 15 spectral candidates
    eval_results = []
    top_configs = scored[:15]

    # Also add some configs near Ng's optimal that spectral might have missed
    extra_configs = [
        {"config": (43, 50), "search_score": 0, "displacement_ratio": 0, "block_size": 7},
        {"config": (44, 51), "search_score": 0, "displacement_ratio": 0, "block_size": 7},
        {"config": (46, 53), "search_score": 0, "displacement_ratio": 0, "block_size": 7},
        {"config": (40, 50), "search_score": 0, "displacement_ratio": 0, "block_size": 10},
        {"config": (42, 52), "search_score": 0, "displacement_ratio": 0, "block_size": 10},
    ]

    all_to_test = top_configs + extra_configs

    for idx, s in enumerate(all_to_test):
        start, end = s["config"]
        bs = end - start

        layer_order = list(range(end)) + list(range(start, end)) + list(range(end, N))
        dup_layers = [original_layers[k] for k in layer_order]
        setattr(inner, attr_name, nn.ModuleList(dup_layers))
        model.config.num_hidden_layers = len(dup_layers)

        result = run_math_probe(gen_fn, verbose=False)
        delta = result['score'] - baseline['score']

        eval_results.append({
            "config": (start, end),
            "score": result['score'],
            "delta": delta,
            "beats_ng": delta > ng_delta,
            "spectral_score": s["search_score"],
        })

        status = "BEATS NG!" if delta > ng_delta else ("better" if delta > 0 else "worse")
        print(f"  [{idx+1:2d}/{len(all_to_test)}] ({start:3d},{end:3d}) bs={bs} "
              f"score={result['score']:.4f} delta={delta:+.4f} [{status}]")

        # Restore
        setattr(inner, attr_name, nn.ModuleList(original_layers))
        model.config.num_hidden_layers = N

    # Summary
    eval_results.sort(key=lambda x: x["delta"], reverse=True)

    print(f"\n{'=' * 60}")
    print("SUMMARY: Spectral-Guided Search vs Ng's Grid Search")
    print(f"{'=' * 60}")
    print(f"Baseline: {baseline['score']:.4f}")
    print(f"Ng (45,52): {ng_result['score']:.4f} (delta: {ng_delta:+.4f})")
    print(f"\nBest found:")
    for r in eval_results[:5]:
        beats = " *** BEATS NG ***" if r["beats_ng"] else ""
        print(f"  ({r['config'][0]:3d},{r['config'][1]:3d}) delta={r['delta']:+.4f}{beats}")

    num_better = sum(1 for r in eval_results if r["delta"] > 0)
    num_beats_ng = sum(1 for r in eval_results if r["beats_ng"])
    print(f"\n{num_better}/{len(eval_results)} configs improved over baseline")
    print(f"{num_beats_ng}/{len(eval_results)} configs beat Ng's (45,52)")
    print(f"Total configs tested: {len(eval_results)} (vs Ng's 3,241 full sweep)")

    # Save
    all_results = {
        "baseline": baseline['score'],
        "ng_config": {"config": [45, 52], "score": ng_result['score'], "delta": ng_delta},
        "spectral_candidates": len(scored),
        "evaluated": [{**r, "config": list(r["config"])} for r in eval_results],
        "num_beats_ng": num_beats_ng,
    }
    with open(RESULTS_DIR / "search_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
