"""
DeepPass Adaptive Depth Inference

Instead of always duplicating a block a fixed number of times, the model
decides per-input how many passes to run based on a convergence criterion.

From TRM / Deep Equilibrium Model theory:
  1. Run layers [0, i) normally
  2. Run block [i, j) on hidden state h to get F(h)
  3. Compute residual: ||F(h) - h|| (hidden state norm change)
  4. If residual > threshold: run the block again (h <- F(h), compute F(h))
  5. Repeat until residual < threshold or max_passes reached
  6. Run layers [j, N) normally
  7. Decode through LM head

Implementation strategy (practical & efficient):
  - For each input prompt, determine the adaptive pass count by measuring
    hidden state convergence on just the prompt (one forward pass per candidate).
  - Once the pass count is chosen, generate all tokens with that fixed config.
  - This is "per-input adaptive" — different questions may get different depths.

Following layer_duplicator.py: we modify the model's layer list via
nn.ModuleList, set use_cache=False, and call model() directly.
"""

import sys
import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe, MATH_QUESTIONS

RESULTS_DIR = Path(__file__).parent.parent / "results"
MODEL_PATH = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"


def set_layer_config(model, original_layers, N, block_start, block_end, n_passes):
    """
    Set the model's layer list for n_passes through block [block_start, block_end).

    Layer sequence: [0..block_end) + [block_start..block_end) * (n_passes-1) + [block_end..N)
    """
    inner = model.model if hasattr(model, 'model') else model.transformer
    attr_name = 'layers' if hasattr(inner, 'layers') else 'h'

    layer_seq = list(original_layers[:block_end])
    for _ in range(n_passes - 1):
        layer_seq.extend(original_layers[block_start:block_end])
    layer_seq.extend(original_layers[block_end:])

    setattr(inner, attr_name, nn.ModuleList(layer_seq))
    model.config.num_hidden_layers = len(layer_seq)


def restore_layers(model, original_layers, N):
    """Restore original layer configuration."""
    inner = model.model if hasattr(model, 'model') else model.transformer
    attr_name = 'layers' if hasattr(inner, 'layers') else 'h'
    setattr(inner, attr_name, nn.ModuleList(original_layers))
    model.config.num_hidden_layers = N


def measure_convergence(model, tokenizer, prompt, original_layers, N,
                        block_start, block_end, max_passes):
    """
    Measure hidden state convergence for a given prompt across pass counts.

    Returns a list of (n_passes, residual_norm) tuples.
    The residual is ||h_after_block(n) - h_after_block(n-1)||.
    """
    inner = model.model if hasattr(model, 'model') else model.transformer
    attr_name = 'layers' if hasattr(inner, 'layers') else 'h'
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    block_size = block_end - block_start

    convergence_data = []
    prev_h = None

    for n_pass in range(1, max_passes + 1):
        # Set up layers for n_pass passes
        set_layer_config(model, original_layers, N, block_start, block_end, n_pass)

        with torch.no_grad():
            outputs = model(input_ids, use_cache=False, output_hidden_states=True)

        # Hidden state index after the adaptive block:
        # Layer 0..block_end-1 is block_end layers, then (n_pass-1)*block_size more
        # output_hidden_states[0] = embedding, [k] = after layer k-1
        h_idx = block_end + (n_pass - 1) * block_size
        h_after = outputs.hidden_states[h_idx].float()  # (batch, seq, dim)

        if prev_h is not None:
            # Residual: L2 norm of the difference, averaged over sequence positions
            residual = torch.norm(h_after - prev_h, dim=-1).mean().item()
            convergence_data.append((n_pass, residual))
        else:
            convergence_data.append((n_pass, float('inf')))

        prev_h = h_after

    # Restore
    restore_layers(model, original_layers, N)
    return convergence_data


def choose_adaptive_passes(convergence_data, threshold):
    """
    Given convergence data [(n_passes, residual), ...], choose the number of
    passes where the residual first drops below threshold.

    If the first pass already has residual below threshold (or max_passes=1),
    return 1. If nothing converges, return the max.
    """
    for n_pass, residual in convergence_data:
        if n_pass == 1:
            continue  # First entry has inf residual (no comparison)
        if residual < threshold:
            return n_pass
    # Never converged — return max
    return convergence_data[-1][0]


def adaptive_generate(model, tokenizer, prompt, original_layers, N,
                      block_start, block_end, threshold, max_passes,
                      pass_tracker=None):
    """
    Per-input adaptive generation:
    1. Measure convergence on the prompt to choose pass count
    2. Generate all tokens with that fixed pass count
    """
    # Step 1: Measure convergence to choose pass count
    convergence_data = measure_convergence(
        model, tokenizer, prompt, original_layers, N,
        block_start, block_end, max_passes
    )
    chosen_passes = choose_adaptive_passes(convergence_data, threshold)

    if pass_tracker is not None:
        pass_tracker["chosen_passes"].append(chosen_passes)
        pass_tracker["convergence_data"].append(convergence_data)

    # Step 2: Generate with chosen pass count
    set_layer_config(model, original_layers, N, block_start, block_end, chosen_passes)
    text = generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    restore_layers(model, original_layers, N)

    return text


def run_fixed_pass(model, tokenizer, original_layers, N, block, n_passes):
    """Run math probe with a fixed n-pass duplication."""
    i, j = block
    set_layer_config(model, original_layers, N, i, j, n_passes)
    gen_fn = lambda prompt: generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    result = run_math_probe(gen_fn, verbose=False)
    restore_layers(model, original_layers, N)
    return result["score"]


def main():
    import sys as _sys
    # Force unbuffered output
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)

    print("=" * 70)
    print("DeepPass Adaptive Depth Inference")
    print("=" * 70)

    run_dir = RESULTS_DIR / f"adaptive_depth_{Path(MODEL_PATH).name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model, tokenizer = load_original_model(MODEL_PATH)
    inner = model.model if hasattr(model, 'model') else model.transformer
    attr_name = 'layers' if hasattr(inner, 'layers') else 'h'
    original_layers = list(getattr(inner, attr_name))
    N = len(original_layers)
    print(f"Model loaded: {N} layers")

    # Parameters
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]
    max_passes_list = [1, 2, 3, 4, 5]
    blocks = [(10, 11), (18, 21), (8, 15)]

    all_results = {}

    # ==== Phase 1: Baseline (no duplication) ====
    print(f"\n{'='*70}")
    print("Phase 1: Baseline (no duplication, 1 pass)")
    print(f"{'='*70}")

    gen_fn = lambda prompt: generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    baseline_result = run_math_probe(gen_fn, verbose=True)
    baseline_score = baseline_result["score"]
    all_results["baseline"] = {"score": baseline_score}
    print(f"\nBaseline score: {baseline_score:.4f}")

    # ==== Phase 2: Fixed 2-pass duplication baselines ====
    print(f"\n{'='*70}")
    print("Phase 2: Fixed 2-pass duplication baselines")
    print(f"{'='*70}")

    fixed_2pass = {}
    for block in blocks:
        i, j = block
        print(f"\n  Block ({i},{j}), fixed 2-pass:")
        score = run_fixed_pass(model, tokenizer, original_layers, N, block, n_passes=2)
        delta = score - baseline_score
        fixed_2pass[f"({i},{j})"] = {"score": score, "delta": delta}
        print(f"    Score: {score:.4f} (delta: {delta:+.4f})")

    all_results["fixed_2pass"] = fixed_2pass

    # ==== Phase 3: Convergence analysis ====
    # Before running the full adaptive sweep, measure convergence on a few
    # representative prompts to understand the residual landscape.
    print(f"\n{'='*70}")
    print("Phase 3: Convergence analysis (residual landscape)")
    print(f"{'='*70}")

    # Use 3 representative math probe prompts
    sample_prompts = []
    for q in MATH_QUESTIONS[:3]:
        from math_probe import USER_TEMPLATE, SYSTEM_PROMPT
        prompt = USER_TEMPLATE.format(question=q["question"])
        full_prompt = f"System: {SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"
        sample_prompts.append(full_prompt)

    convergence_analysis = {}
    for block in blocks:
        i, j = block
        block_key = f"({i},{j})"
        convergence_analysis[block_key] = {}

        for pidx, prompt in enumerate(sample_prompts):
            conv = measure_convergence(
                model, tokenizer, prompt, original_layers, N,
                i, j, max_passes=5
            )
            convergence_analysis[block_key][f"prompt_{pidx}"] = conv
            residuals = [r for _, r in conv if r != float('inf')]
            print(f"  Block {block_key}, prompt {pidx}: "
                  f"residuals = {['inf'] + [f'{r:.2f}' for r in residuals]}")

    all_results["convergence_analysis"] = {
        bk: {pk: [(n, r if r != float('inf') else "inf") for n, r in data]
             for pk, data in pdata.items()}
        for bk, pdata in convergence_analysis.items()
    }

    # ==== Phase 4: Adaptive depth sweep ====
    print(f"\n{'='*70}")
    print("Phase 4: Adaptive depth sweep")
    print(f"{'='*70}")

    adaptive_results = {}

    for block in blocks:
        i, j = block
        block_key = f"({i},{j})"
        adaptive_results[block_key] = {}

        for max_passes in max_passes_list:
            for threshold in thresholds:
                config_key = f"thresh={threshold}_maxpass={max_passes}"

                # max_passes=1 ignores threshold (always 1 pass = baseline)
                if max_passes == 1 and threshold != thresholds[0]:
                    continue

                print(f"\n  Block {block_key}, max_passes={max_passes}, threshold={threshold}")

                pass_tracker = {
                    "chosen_passes": [],
                    "convergence_data": [],
                }

                def make_gen_fn(blk_i, blk_j, thresh, mp, tracker):
                    def fn(prompt):
                        return adaptive_generate(
                            model, tokenizer, prompt, original_layers, N,
                            block_start=blk_i, block_end=blk_j,
                            threshold=thresh, max_passes=mp,
                            pass_tracker=tracker,
                        )
                    return fn

                gen_fn = make_gen_fn(i, j, threshold, max_passes, pass_tracker)
                t0 = time.time()
                result = run_math_probe(gen_fn, verbose=True)
                elapsed = time.time() - t0

                chosen_list = pass_tracker["chosen_passes"]
                avg_passes = sum(chosen_list) / len(chosen_list) if chosen_list else 1.0

                # Distribution of chosen passes
                pass_distribution = {}
                for p in range(1, max_passes + 1):
                    pass_distribution[str(p)] = chosen_list.count(p)

                score = result["score"]
                delta = score - baseline_score

                entry = {
                    "score": score,
                    "delta": delta,
                    "avg_passes": avg_passes,
                    "pass_distribution": pass_distribution,
                    "per_input_passes": chosen_list,
                    "elapsed_s": elapsed,
                    "threshold": threshold,
                    "max_passes": max_passes,
                }
                adaptive_results[block_key][config_key] = entry

                print(f"    Score: {score:.4f} (delta: {delta:+.4f})")
                print(f"    Avg passes: {avg_passes:.2f}")
                print(f"    Per-input: {chosen_list}")
                print(f"    Distribution: {pass_distribution}")
                print(f"    Time: {elapsed:.1f}s")

    all_results["adaptive"] = adaptive_results

    # ==== Summary ====
    print(f"\n{'='*70}")
    print("ADAPTIVE DEPTH SUMMARY")
    print(f"{'='*70}")
    print(f"\nBaseline (no duplication): {baseline_score:.4f}")

    print(f"\n--- Fixed 2-pass baselines ---")
    print(f"{'Block':>12} {'Score':>8} {'Delta':>10}")
    for bk, bv in fixed_2pass.items():
        print(f"{bk:>12} {bv['score']:8.4f} {bv['delta']:+10.4f}")

    print(f"\n--- Adaptive depth results ---")
    header = (f"{'Block':>12} {'MaxPass':>8} {'Thresh':>8} {'Score':>8} "
              f"{'Delta':>10} {'AvgPass':>8} {'Adapts?':>8}")
    print(header)

    for block_key in adaptive_results:
        for config_key, entry in adaptive_results[block_key].items():
            mp = entry["max_passes"]
            ap = entry["avg_passes"]
            if mp == 1:
                adapts = "N/A"
            elif ap > 1.0 and ap < float(mp):
                adapts = "YES"
            else:
                adapts = "NO"
            print(f"{block_key:>12} {mp:>8} {entry['threshold']:>8.1f} "
                  f"{entry['score']:>8.4f} {entry['delta']:>+10.4f} "
                  f"{ap:>8.2f} {adapts:>8}")

    # Best adaptive vs best fixed per block
    print(f"\n--- Comparison: Best adaptive vs fixed 2-pass ---")
    for block_key in adaptive_results:
        fixed_score = fixed_2pass.get(block_key, {}).get("score", baseline_score)
        best_adaptive = max(
            adaptive_results[block_key].values(),
            key=lambda x: x["score"]
        )
        print(f"\n  Block {block_key}:")
        print(f"    Fixed 2-pass:    {fixed_score:.4f}")
        print(f"    Best adaptive:   {best_adaptive['score']:.4f} "
              f"(max_passes={best_adaptive['max_passes']}, "
              f"threshold={best_adaptive['threshold']}, "
              f"avg_passes={best_adaptive['avg_passes']:.2f})")
        diff = best_adaptive['score'] - fixed_score
        if diff > 0.001:
            print(f"    => Adaptive WINS by {diff:+.4f}")
        elif diff < -0.001:
            print(f"    => Fixed WINS by {-diff:+.4f}")
        else:
            print(f"    => TIE (within 0.001)")

    # Save
    with open(run_dir / "adaptive_depth_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {run_dir / 'adaptive_depth_results.json'}")


if __name__ == "__main__":
    main()
