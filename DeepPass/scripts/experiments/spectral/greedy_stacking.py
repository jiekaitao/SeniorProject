"""
DeepPass Greedy Iterative Layer Stacking

Novel experiment: instead of choosing multiple blocks simultaneously (which causes
interference), choose blocks GREEDILY — pick the best block, apply it, then find
the next best block on the MODIFIED model's dynamics.

Algorithm:
  Iteration 1: Spectral screen original model → evaluate top-K → pick best (i1,j1)
  Iteration 2: Spectral screen MODIFIED model → evaluate top-K → pick best (i2,j2)
  Stop when score doesn't improve.

This is distinct from both multi-block (simultaneous) and multi-pass (same block repeated).
The key insight is that spectral screening on the modified model sees DIFFERENT dynamics,
potentially finding complementary blocks that stack instead of interfere.
"""

import json
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Add scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe


SPECTRAL_PROMPTS = [
    "The theory of general relativity describes",
    "In mathematics, a topological space is",
    "def fibonacci(n):\n    if n <= 1:",
    "What is 123 multiplied by 456?",
]


def spectral_screen_on_model(
    model, tokenizer, prompts, step, block_sizes, exclude_ranges=None
):
    """
    Compute displacement_rho for each candidate block on the CURRENT model state.

    Runs on whatever model state is currently set (possibly modified with previous
    blocks). For each candidate, temporarily adds another duplication on top,
    computes the displacement metric, then restores to the current state.

    Args:
        model: HF model (possibly already modified with previous blocks)
        tokenizer: tokenizer
        prompts: list of prompt strings for spectral analysis
        step: step size for start position sweep
        block_sizes: list of block sizes to try
        exclude_ranges: list of (start, end) ranges to skip (duplicate copy positions)

    Returns:
        List of dicts sorted by displacement_rho (ascending = most contractive first),
        each with {start, end, block_size, displacement_rho}
    """
    inner = model.model if hasattr(model, "model") else model.transformer
    attr_name = "layers" if hasattr(inner, "layers") else "h"
    current_layers = list(getattr(inner, attr_name))
    N = len(current_layers)
    device = next(model.parameters()).device

    if exclude_ranges is None:
        exclude_ranges = []

    # Generate candidate blocks
    candidates = []
    for bs in block_sizes:
        for start in range(0, N - bs + 1, step):
            end = start + bs
            # Check overlap with exclude ranges
            skip = False
            for ex_s, ex_e in exclude_ranges:
                if not (end <= ex_s or start >= ex_e):
                    skip = True
                    break
            if skip:
                continue
            candidates.append((start, end))

    print(
        f"  Spectral screening {len(candidates)} candidates "
        f"(N={N}, step={step}, block_sizes={block_sizes}, "
        f"exclude={exclude_ranges})"
    )

    results = []
    for idx, (start, end) in enumerate(candidates):
        disp_rhos = []

        for prompt in prompts:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=128
            ).to(device)

            with torch.no_grad():
                # Normal forward pass (current model state)
                out_normal = model(**inputs, use_cache=False)
                logits_normal = out_normal.logits.detach()

                # Duplicated forward pass: add one more duplication of [start, end)
                layer_order = (
                    list(range(end))
                    + list(range(start, end))
                    + list(range(end, N))
                )
                dup_layers = [current_layers[k] for k in layer_order]
                setattr(inner, attr_name, nn.ModuleList(dup_layers))
                model.config.num_hidden_layers = len(dup_layers)

                out_dup = model(**inputs, use_cache=False)
                logits_dup = out_dup.logits.detach()

                # Restore current state
                setattr(inner, attr_name, nn.ModuleList(current_layers))
                model.config.num_hidden_layers = N

                # Compute displacement
                diff_norm = (
                    (logits_dup - logits_normal).float().norm(dim=-1).mean().item()
                )
                logit_norm = logits_normal.float().norm(dim=-1).mean().item()

                if logit_norm > 1e-10:
                    disp_rhos.append(diff_norm / logit_norm)

        disp_rho = float(np.mean(disp_rhos)) if disp_rhos else float("inf")
        results.append(
            {
                "start": start,
                "end": end,
                "block_size": end - start,
                "displacement_rho": disp_rho,
            }
        )

        if (idx + 1) % 10 == 0 or idx == len(candidates) - 1:
            print(
                f"    [{idx+1}/{len(candidates)}] "
                f"({start},{end}) disp_rho={disp_rho:.4f}"
            )

    # Sort by displacement_rho (low = more contractive = better candidate)
    results.sort(key=lambda x: x["displacement_rho"])
    return results


def build_greedy_layer_order(blocks, original_N):
    """
    Build the full layer execution order by applying blocks sequentially.

    Each block (i, j) is in the index space of the model AFTER all previous
    blocks have been applied. The returned list contains indices into the
    ORIGINAL layer list.

    Args:
        blocks: list of (i, j) tuples
        original_N: number of layers in the original model

    Returns:
        List of indices into the original layer list
    """
    # Start with identity mapping
    current_order = list(range(original_N))

    for i, j in blocks:
        N = len(current_order)
        assert 0 <= i < j <= N, f"Invalid block ({i},{j}) for model with {N} layers"
        # Insert duplicate of [i,j) right after position j-1
        new_order = current_order[:j] + current_order[i:j] + current_order[j:]
        current_order = new_order

    return current_order


def apply_greedy_blocks(model, blocks, original_layers, original_N):
    """
    Apply the full greedy block sequence to the model.

    Starts from original_layers and applies each block sequentially,
    each in the index space of the model after previous blocks.

    Returns:
        (new_layers_list, new_N)
    """
    inner = model.model if hasattr(model, "model") else model.transformer
    attr_name = "layers" if hasattr(inner, "layers") else "h"

    layer_order = build_greedy_layer_order(blocks, original_N)
    new_layers = [original_layers[k] for k in layer_order]

    setattr(inner, attr_name, nn.ModuleList(new_layers))
    model.config.num_hidden_layers = len(new_layers)

    return new_layers, len(new_layers)


def restore_model(model, original_layers, original_N):
    """Restore model to original state."""
    inner = model.model if hasattr(model, "model") else model.transformer
    attr_name = "layers" if hasattr(inner, "layers") else "h"
    setattr(inner, attr_name, nn.ModuleList(original_layers))
    model.config.num_hidden_layers = original_N


def evaluate_with_blocks(
    model, tokenizer, blocks, original_layers, original_N, dual_probe=False
):
    """
    Evaluate the model with the given block sequence using math_probe
    and optionally EQ-Bench probe.

    Applies all blocks from scratch (from original model state), runs
    probes, then restores model to original. This avoids accumulated
    state bugs from sequential modifications.

    Returns:
        dict with 'math_score', and optionally 'eq_score' and 'combined_score'
    """
    try:
        apply_greedy_blocks(model, blocks, original_layers, original_N)

        def gen_fn(prompt):
            return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)

        math_result = run_math_probe(gen_fn, verbose=False)
        result = {"math_score": math_result["score"]}

        if dual_probe:
            # EQ-Bench needs longer responses for emotion scoring
            def gen_fn_long(prompt):
                return generate_no_cache(
                    model, tokenizer, prompt, max_new_tokens=128
                )

            eq_result = run_eq_bench_probe(gen_fn_long, verbose=False)
            result["eq_score"] = eq_result["score"]
            result["eq_parse_rate"] = eq_result["parse_rate"]
            # Combined: math_score (0-1) * 50 + eq_score (0-100) * 0.5
            # Both contribute equally on a 0-50 scale, total 0-100
            result["combined_score"] = (
                result["math_score"] * 50 + result["eq_score"] * 0.5
            )

        return result
    finally:
        restore_model(model, original_layers, original_N)


def compute_exclude_ranges(selected_blocks, original_N):
    """
    Compute which positions in the modified model are duplicate copies.

    After applying blocks sequentially, tracks where duplicates are inserted.
    Returns ranges in the FINAL modified model's index space that should be
    excluded from future spectral screening.

    Example: blocks [(10,11), (5,8)] on 28-layer model
      After (10,11): 29 layers, duplicate at [11,12)
      After (5,8): 32 layers, new duplicate at [8,11), old shifted to [14,15)
      Returns: [(8,11), (14,15)]
    """
    exclude = []

    for i, j in selected_blocks:
        block_size = j - i
        # Shift existing exclude ranges that are at or after position j
        updated = []
        for s, e in exclude:
            if s >= j:
                updated.append((s + block_size, e + block_size))
            elif e > j:
                # Range spans the insertion point
                updated.append((s, j))
                updated.append((j + block_size, e + block_size))
            else:
                updated.append((s, e))
        # Add the new duplicate range: copies are at [j, j + block_size)
        updated.append((j, j + block_size))
        exclude = updated

    return exclude


def save_results(
    output_dir, baseline_score, scores, selected_blocks, iteration_results, original_N,
    math_scores=None, eq_scores=None, dual_probe=False,
):
    """Save results to JSON."""
    output_dir = Path(output_dir)

    results = {
        "experiment": "greedy_stacking",
        "timestamp": datetime.now().isoformat(),
        "dual_probe": dual_probe,
        "baseline_score": baseline_score,
        "selected_blocks": [list(b) for b in selected_blocks],
        "scores": scores,
        "final_score": scores[-1],
        "total_improvement": scores[-1] - baseline_score,
        "original_N": original_N,
        "final_N": original_N + sum(j - i for i, j in selected_blocks),
        "layer_order": build_greedy_layer_order(selected_blocks, original_N),
        "iterations": iteration_results,
    }
    if math_scores:
        results["math_scores"] = math_scores
    if eq_scores:
        results["eq_scores"] = eq_scores

    with open(output_dir / "greedy_stacking_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Results saved to {output_dir / 'greedy_stacking_results.json'}")


def run_greedy_stacking(
    model_path: str,
    max_iterations: int = 3,
    top_k: int = 5,
    step: int = 2,
    block_sizes: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    force: bool = False,
    dual_probe: bool = False,
):
    """
    Main greedy stacking loop.

    For each iteration:
      1. Spectral screen the (possibly modified) model to rank candidate blocks
      2. Evaluate top-K candidates with math_probe (+ EQ-Bench if dual_probe)
      3. Select the best block (or least-bad if --force)
      4. Apply and continue to next iteration

    Args:
        model_path: HuggingFace model path
        max_iterations: maximum number of blocks to stack
        top_k: number of candidates to evaluate per iteration
        step: step size for spectral screening sweep
        block_sizes: list of block sizes to try
        output_dir: directory to save results
        force: if True, always pick the best candidate even if score decreases
        dual_probe: if True, also run EQ-Bench probe and use combined score
    """
    if output_dir is None:
        output_dir = (
            "/blue/cis4914/jietao/DeepPass/results/greedy_stacking/"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if block_sizes is None:
        block_sizes = [1, 2, 3, 5, 7]

    print("=" * 70)
    print("DeepPass Greedy Iterative Layer Stacking")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Max iterations: {max_iterations}")
    print(f"Top-K per iteration: {top_k}")
    print(f"Block sizes: {block_sizes}")
    print(f"Step: {step}")
    print(f"Force continue: {force}")
    print(f"Dual probe (math + EQ-Bench): {dual_probe}")
    print(f"Output: {output_dir}")
    print()

    # Load model
    model, tokenizer = load_original_model(model_path)
    model.eval()

    inner = model.model if hasattr(model, "model") else model.transformer
    attr_name = "layers" if hasattr(inner, "layers") else "h"
    original_layers = list(getattr(inner, attr_name))
    original_N = len(original_layers)

    # Baseline
    print("--- Baseline (no duplication) ---")
    t0 = time.time()

    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)

    baseline = run_math_probe(gen_fn, verbose=True)
    baseline_math = baseline["score"]

    baseline_eq = None
    baseline_combined = None
    if dual_probe:
        print("\n--- Baseline EQ-Bench ---")

        def gen_fn_long(prompt):
            return generate_no_cache(model, tokenizer, prompt, max_new_tokens=128)

        eq_baseline = run_eq_bench_probe(gen_fn_long, verbose=True)
        baseline_eq = eq_baseline["score"]
        baseline_combined = baseline_math * 50 + baseline_eq * 0.5
        print(
            f"Baseline combined: {baseline_combined:.2f} "
            f"(math={baseline_math:.4f}, eq={baseline_eq:.1f}/100)"
        )

    # Primary score used for selection: combined if dual, math if single
    baseline_score = baseline_combined if dual_probe else baseline_math
    print(f"Baseline score: {baseline_score:.4f} ({time.time()-t0:.1f}s)\n")

    selected_blocks = []
    scores = [baseline_score]
    math_scores = [baseline_math]
    eq_scores = [baseline_eq] if dual_probe else []
    iteration_results = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}")

        iter_t0 = time.time()

        # Apply all previously selected blocks for screening
        if selected_blocks:
            apply_greedy_blocks(model, selected_blocks, original_layers, original_N)
            current_N = model.config.num_hidden_layers
            print(
                f"Model state: {len(selected_blocks)} blocks applied, "
                f"{current_N} layers"
            )
        else:
            current_N = original_N
            print(f"Model state: original, {current_N} layers")

        # Compute exclude ranges (duplicate copy positions in current model)
        exclude_ranges = (
            compute_exclude_ranges(selected_blocks, original_N)
            if selected_blocks
            else []
        )

        # Phase 1: Spectral screening
        print(f"\n--- Phase 1: Spectral Screening ---")
        t0 = time.time()
        candidates = spectral_screen_on_model(
            model,
            tokenizer,
            SPECTRAL_PROMPTS,
            step=step,
            block_sizes=block_sizes,
            exclude_ranges=exclude_ranges,
        )
        screen_time = time.time() - t0
        print(f"Screening done in {screen_time:.1f}s")

        # Restore model before evaluation phase
        restore_model(model, original_layers, original_N)

        if not candidates:
            print(f"\n--- STOPPING: No candidates available at iteration {iteration} ---")
            break

        # Show top candidates
        show_n = min(top_k * 2, len(candidates))
        print(f"\nTop {show_n} candidates by displacement_rho:")
        for i, c in enumerate(candidates[:show_n]):
            print(
                f"  {i+1:2d}. ({c['start']:3d},{c['end']:3d}) "
                f"bs={c['block_size']} disp_rho={c['displacement_rho']:.4f}"
            )

        # Phase 2: Evaluate top-K with probes
        eval_k = min(top_k, len(candidates))
        probe_label = "Dual Probe" if dual_probe else "Math Probe"
        print(f"\n--- Phase 2: {probe_label} Evaluation (top-{eval_k}) ---")
        best_score = -float("inf")
        best_block = None
        best_eval_result = None
        eval_results = []

        for rank, candidate in enumerate(candidates[:eval_k]):
            block = (candidate["start"], candidate["end"])
            test_blocks = selected_blocks + [block]

            t0 = time.time()
            eval_out = evaluate_with_blocks(
                model, tokenizer, test_blocks, original_layers, original_N,
                dual_probe=dual_probe,
            )
            eval_time = time.time() - t0

            # Primary score for selection
            if dual_probe:
                score = eval_out["combined_score"]
            else:
                score = eval_out["math_score"]

            delta = score - scores[-1]
            total_delta = score - baseline_score

            result = {
                "rank": rank + 1,
                "block": list(block),
                "score": score,
                "math_score": eval_out["math_score"],
                "delta_from_prev": delta,
                "delta_from_baseline": total_delta,
                "displacement_rho": candidate["displacement_rho"],
                "eval_time": eval_time,
            }
            if dual_probe:
                result["eq_score"] = eval_out["eq_score"]
                result["eq_parse_rate"] = eval_out["eq_parse_rate"]
                result["combined_score"] = eval_out["combined_score"]
            eval_results.append(result)

            indicator = " ***BEST***" if score > best_score else ""
            if dual_probe:
                print(
                    f"  [{rank+1}/{eval_k}] ({block[0]:3d},{block[1]:3d}) "
                    f"combined={score:.2f} math={eval_out['math_score']:.4f} "
                    f"eq={eval_out['eq_score']:.1f} "
                    f"delta={delta:+.2f}{indicator} [{eval_time:.1f}s]"
                )
            else:
                print(
                    f"  [{rank+1}/{eval_k}] ({block[0]:3d},{block[1]:3d}) "
                    f"score={score:.4f} delta={delta:+.4f} "
                    f"total_delta={total_delta:+.4f}{indicator} [{eval_time:.1f}s]"
                )

            if score > best_score:
                best_score = score
                best_block = block
                best_eval_result = eval_out

        iter_time = time.time() - iter_t0
        improved = best_score > scores[-1]

        # Record iteration results
        iter_result = {
            "iteration": iteration,
            "model_layers": current_N,
            "num_candidates_screened": len(candidates),
            "top_candidates": candidates[: top_k * 2],
            "evaluations": eval_results,
            "best_block": list(best_block) if best_block else None,
            "best_score": best_score,
            "previous_score": scores[-1],
            "improvement": best_score - scores[-1],
            "total_delta": best_score - baseline_score,
            "improved": improved,
            "exclude_ranges": exclude_ranges,
            "screening_time": screen_time,
            "time_seconds": iter_time,
        }
        iteration_results.append(iter_result)

        # Save intermediate results
        save_results(
            output_dir,
            baseline_score,
            scores + ([best_score] if best_block else []),
            selected_blocks + ([best_block] if best_block else []),
            iteration_results,
            original_N,
            math_scores=math_scores,
            eq_scores=eq_scores if dual_probe else None,
            dual_probe=dual_probe,
        )

        # Check termination (unless --force)
        if best_block is None:
            print(f"\n--- STOPPING: No candidates available ---")
            break

        if not improved and not force:
            print(f"\n--- STOPPING: No improvement at iteration {iteration} ---")
            print(
                f"Previous score: {scores[-1]:.4f}, "
                f"Best candidate: {best_score:.4f}"
            )
            print("Use --force to continue through negative deltas")
            break

        # Accept the best block (even if negative delta when --force)
        selected_blocks.append(best_block)
        scores.append(best_score)
        math_scores.append(best_eval_result["math_score"])
        if dual_probe:
            eq_scores.append(best_eval_result["eq_score"])
        direction = "+" if improved else ""
        status = "IMPROVED" if improved else "DECLINED (forced)"
        print(f"\n--- {status}: block ({best_block[0]},{best_block[1]}) ---")
        print(
            f"Score: {scores[-2]:.4f} -> {best_score:.4f} "
            f"({direction}{best_score - scores[-2]:.4f})"
        )
        if dual_probe:
            print(
                f"Math: {math_scores[-2]:.4f} -> {math_scores[-1]:.4f}  "
                f"EQ: {eq_scores[-2]:.1f} -> {eq_scores[-1]:.1f}"
            )
        print(f"Iteration time: {iter_time:.1f}s")

    # Final summary
    print(f"\n{'='*70}")
    print("GREEDY STACKING RESULTS")
    print(f"{'='*70}")
    print(f"Baseline: {baseline_score:.4f}")
    print(f"Blocks selected: {len(selected_blocks)}")
    peak_score = baseline_score
    peak_iter = 0
    for i, ((bi, bj), score) in enumerate(zip(selected_blocks, scores[1:])):
        prev = scores[i]
        delta = score - prev
        direction = "+" if delta >= 0 else ""
        marker = ""
        if score > peak_score:
            peak_score = score
            peak_iter = i + 1
            marker = " <-- PEAK"
        extra = ""
        if dual_probe:
            extra = f" [math={math_scores[i+1]:.4f} eq={eq_scores[i+1]:.1f}]"
        print(
            f"  Iteration {i+1}: ({bi},{bj}) -> score={score:.4f} "
            f"({direction}{delta:.4f} from prev, "
            f"+{score - baseline_score:.4f} from baseline){extra}{marker}"
        )
    if peak_iter > 0:
        print(f"\nPeak at iteration {peak_iter}: {peak_score:.4f}")
    print(f"Final score: {scores[-1]:.4f}")
    print(f"Total improvement: {scores[-1] - baseline_score:+.4f}")

    if selected_blocks:
        layer_order = build_greedy_layer_order(selected_blocks, original_N)
        print(f"Final layer count: {len(layer_order)} (original: {original_N})")
        print(f"Layer order: {layer_order}")

    print(f"{'='*70}")

    # Save final results
    save_results(
        output_dir,
        baseline_score,
        scores,
        selected_blocks,
        iteration_results,
        original_N,
        math_scores=math_scores,
        eq_scores=eq_scores if dual_probe else None,
        dual_probe=dual_probe,
    )

    return {
        "baseline_score": baseline_score,
        "selected_blocks": selected_blocks,
        "scores": scores,
        "iterations": iteration_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepPass Greedy Iterative Layer Stacking"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct",
    )
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument(
        "--block-sizes",
        type=str,
        default="1,2,3,5,7",
        help="Comma-separated block sizes to try",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Continue through negative deltas, always picking the least-bad candidate",
    )
    parser.add_argument(
        "--dual-probe",
        action="store_true",
        help="Use both math_probe and EQ-Bench probe (combined score for selection)",
    )
    args = parser.parse_args()

    block_sizes = [int(x) for x in args.block_sizes.split(",")]

    run_greedy_stacking(
        model_path=args.model,
        max_iterations=args.max_iterations,
        top_k=args.top_k,
        step=args.step,
        block_sizes=block_sizes,
        output_dir=args.output,
        force=args.force,
        dual_probe=args.dual_probe,
    )
