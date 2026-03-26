"""
Corrected Pipeline: Spectral Screen → Dual Probe → Greedy Stack with Adapters

The pipeline that should beat Ng's single-block result:
1. Spectral screening → 20 candidates
2. Dual probe (math + EQ-bench) → pick best block A
3. Apply A with adapter junction
4. Spectral screen modified model → 20 new candidates
5. Dual probe → pick best complementary block B
6. If A+B > A → accept. Else stop.

Uses eq_bench_probe.py (lightweight, ~60s on 7B) alongside math_probe.
"""

import sys, os, json, time, torch, gc, argparse
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', 'scripts'))

from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/corrected_pipeline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPECTRAL_PROMPTS = [
    "The theory of general relativity states that",
    "In Python, a decorator is a function that",
    "What is 78313 multiplied by 88537?",
    "A linked list is a data structure where",
]


# =============================================================================
# Adapter (identity init, dtype-agnostic)
# =============================================================================

class JunctionAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=256, init_scale=0.001):
        super().__init__()
        self.down_weight = nn.Parameter(torch.randn(bottleneck_dim, hidden_dim) * 0.02)
        self.up_weight = nn.Parameter(torch.randn(hidden_dim, bottleneck_dim) * init_scale)

    def forward(self, x):
        dtype = x.dtype
        d = self.down_weight.to(dtype)
        u = self.up_weight.to(dtype)
        return x + nn.functional.linear(
            nn.functional.gelu(nn.functional.linear(x, d)), u
        )


class AdapterWrappedLayer(nn.Module):
    def __init__(self, original_layer, adapter):
        super().__init__()
        self.layer = original_layer
        self.adapter = adapter

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, *args, **kwargs):
        output = self.layer(*args, **kwargs)
        if isinstance(output, tuple):
            h = output[0]
            h = self.adapter(h)
            return (h,) + output[1:]
        return self.adapter(output)


# =============================================================================
# Helpers
# =============================================================================

def _run_layers(inner, h, start, end, pos_embeds):
    for i in range(start, end):
        out = inner.layers[i](h, position_embeddings=pos_embeds, use_cache=False)
        h = out[0] if isinstance(out, tuple) else out
    return h


def spectral_screen(model, tokenizer, step=2, block_sizes=None, exclude=None):
    """Spectral screening in original index space."""
    if block_sizes is None:
        block_sizes = [1, 2, 3, 5]
    if exclude is None:
        exclude = []

    device = next(model.parameters()).device
    inner = model.model
    N = len(inner.layers)
    original_layers = list(inner.layers)

    candidates = []
    for bs in block_sizes:
        for start in range(0, N - bs, step):
            end = start + bs
            if end > N:
                continue
            skip = any(not (end <= es or start >= ee) for es, ee in exclude)
            if skip:
                continue

            disps = []
            for prompt in SPECTRAL_PROMPTS:
                inp = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    h = inner.embed_tokens(inp["input_ids"])
                    seq_len = h.shape[1]
                    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                    pos_embeds = inner.rotary_emb(h, pos_ids)
                    h_base = _run_layers(inner, h, 0, N, pos_embeds)
                    logits_base = model.lm_head(inner.norm(h_base))

                    dup_order = list(range(end)) + list(range(start, end)) + list(range(end, N))
                    inner.layers = nn.ModuleList([original_layers[idx] for idx in dup_order])
                    model.config.num_hidden_layers = len(dup_order)

                    h = inner.embed_tokens(inp["input_ids"])
                    h_dup = _run_layers(inner, h, 0, len(dup_order), pos_embeds)
                    logits_dup = model.lm_head(inner.norm(h_dup))

                    inner.layers = nn.ModuleList(original_layers)
                    model.config.num_hidden_layers = N

                    diff = (logits_dup - logits_base).float()
                    disp = diff.norm() / (logits_base.float().norm() + 1e-8)
                    disps.append(disp.item())

            candidates.append({
                "start": start, "end": end, "block_size": bs,
                "displacement_rho": np.mean(disps),
            })

    candidates.sort(key=lambda x: x["displacement_rho"])
    return candidates


def dual_probe_eval(model, tokenizer, label=""):
    """Run both math probe and EQ-bench probe. Returns combined score."""
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    def gen_fn_long(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=128)

    math_result = run_math_probe(gen_fn, verbose=False)
    eq_result = run_eq_bench_probe(gen_fn_long, verbose=False)

    # Combined: math (0-1) * 50 + eq (0-100) * 0.5 → both on 0-50 scale
    combined = math_result["score"] * 50 + eq_result["score"] * 0.5

    print(f"  {label}: math={math_result['score']:.4f} eq={eq_result['score']:.1f} "
          f"combined={combined:.2f}")

    return {
        "math_score": math_result["score"],
        "eq_score": eq_result["score"],
        "eq_parse_rate": eq_result["parse_rate"],
        "combined": combined,
    }


def apply_blocks_with_adapters(model, blocks_adapters, original_layers, original_N):
    """Apply blocks with adapters. blocks_adapters = [(start, end, adapter), ...]"""
    inner = model.model
    sorted_ba = sorted(blocks_adapters, key=lambda x: x[0])

    order = []
    adapters_at = {}
    prev_j = 0
    for (i, j, adapter) in sorted_ba:
        order.extend(list(range(prev_j, j)))
        order.extend(list(range(i, j)))
        if adapter is not None:
            adapters_at[len(order) - 1] = adapter
        prev_j = j
    order.extend(list(range(prev_j, original_N)))

    new_layers = []
    for idx, layer_idx in enumerate(order):
        layer = original_layers[layer_idx]
        if idx in adapters_at:
            layer = AdapterWrappedLayer(layer, adapters_at[idx])
        new_layers.append(layer)

    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)


def restore(model, original_layers, original_N):
    model.model.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = original_N


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(model_path, top_k=10, max_iterations=3, step=2,
                 block_sizes=None, adapter_bottleneck=256):
    if block_sizes is None:
        block_sizes = [1, 2, 3, 5]

    output_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("CORRECTED PIPELINE: Spectral → Dual Probe → Greedy + Adapters")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Top-K per iteration: {top_k}")
    print(f"Max iterations: {max_iterations}")
    print(f"Block sizes: {block_sizes}")

    t0 = time.time()
    model, tokenizer = load_original_model(model_path)
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    inner = model.model
    original_layers = list(inner.layers)
    original_N = len(original_layers)

    # Baseline
    print(f"\n--- Baseline Dual Probe ---")
    baseline = dual_probe_eval(model, tokenizer, "Baseline")
    print(f"  Baseline combined: {baseline['combined']:.2f}")

    selected_blocks = []  # [(start, end, adapter), ...]
    scores = [baseline["combined"]]
    all_results = {"baseline": baseline, "iterations": []}

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}")

        # Restore model to original for spectral screening
        restore(model, original_layers, original_N)

        # Step 1: Spectral screen
        exclude = [(b[0], b[1]) for b in selected_blocks]
        print(f"\n--- Step 1: Spectral Screening (exclude {exclude}) ---")
        candidates = spectral_screen(model, tokenizer, step=step,
                                     block_sizes=block_sizes, exclude=exclude)
        print(f"  {len(candidates)} candidates, top-5:")
        for c in candidates[:5]:
            print(f"    ({c['start']},{c['end']}) disp_rho={c['displacement_rho']:.4f}")

        # Step 2: Dual probe on top-K
        eval_k = min(top_k, len(candidates))
        print(f"\n--- Step 2: Dual Probe on top-{eval_k} ---")

        best_combined = -float('inf')
        best_block = None
        best_adapter = None
        eval_results = []

        for rank, cand in enumerate(candidates[:eval_k]):
            block = (cand["start"], cand["end"])
            adapter = JunctionAdapter(hidden_dim, adapter_bottleneck).to(device)

            # Apply ALL selected blocks + this candidate
            test_blocks = selected_blocks + [(block[0], block[1], adapter)]
            restore(model, original_layers, original_N)
            apply_blocks_with_adapters(model, test_blocks, original_layers, original_N)

            result = dual_probe_eval(model, tokenizer,
                                     f"[{rank+1}/{eval_k}] ({block[0]},{block[1]})")
            result["block"] = list(block)
            result["displacement_rho"] = cand["displacement_rho"]
            eval_results.append(result)

            if result["combined"] > best_combined:
                best_combined = result["combined"]
                best_block = block
                best_adapter = JunctionAdapter(hidden_dim, adapter_bottleneck).to(device)
                best_adapter.load_state_dict(adapter.state_dict())

            restore(model, original_layers, original_N)

        # Check improvement
        improved = best_combined > scores[-1]
        delta = best_combined - scores[-1]

        iter_result = {
            "iteration": iteration,
            "best_block": list(best_block) if best_block else None,
            "best_combined": best_combined,
            "improved": improved,
            "delta": delta,
            "evaluations": eval_results,
        }
        all_results["iterations"].append(iter_result)

        if not improved:
            print(f"\n--- STOPPING: No improvement ---")
            print(f"  Previous: {scores[-1]:.2f}, Best candidate: {best_combined:.2f}")
            break

        selected_blocks.append((best_block[0], best_block[1], best_adapter))
        scores.append(best_combined)
        print(f"\n--- ACCEPTED: ({best_block[0]},{best_block[1]}) + adapter ---")
        print(f"  Combined: {scores[-2]:.2f} -> {best_combined:.2f} (+{delta:.2f})")

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Baseline: math={baseline['math_score']:.4f} eq={baseline['eq_score']:.1f} "
          f"combined={baseline['combined']:.2f}")
    for i, ((s, e, _), score) in enumerate(zip(selected_blocks, scores[1:])):
        delta = score - scores[i]
        print(f"  + ({s},{e}) with adapter -> combined={score:.2f} (+{delta:.2f})")
    print(f"\nFinal combined: {scores[-1]:.2f} (delta: {scores[-1]-scores[0]:+.2f})")
    print(f"Blocks: {[(b[0], b[1]) for b in selected_blocks]}")
    print(f"Total time: {elapsed/60:.1f} min")

    # Run final dual probe on best config
    if selected_blocks:
        print(f"\n--- Final verification ---")
        restore(model, original_layers, original_N)
        apply_blocks_with_adapters(model, selected_blocks, original_layers, original_N)
        final = dual_probe_eval(model, tokenizer, "FINAL")
        all_results["final"] = final
        restore(model, original_layers, original_N)

    all_results["selected_blocks"] = [(b[0], b[1]) for b in selected_blocks]
    all_results["scores"] = scores
    all_results["elapsed_minutes"] = elapsed / 60

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {output_dir}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--block-sizes", type=str, default="1,2,3,5")
    parser.add_argument("--adapter-bottleneck", type=int, default=256)
    args = parser.parse_args()

    block_sizes = [int(x) for x in args.block_sizes.split(",")]
    run_pipeline(
        model_path=args.model,
        top_k=args.top_k,
        max_iterations=args.max_iterations,
        step=args.step,
        block_sizes=block_sizes,
        adapter_bottleneck=args.adapter_bottleneck,
    )


if __name__ == "__main__":
    main()
