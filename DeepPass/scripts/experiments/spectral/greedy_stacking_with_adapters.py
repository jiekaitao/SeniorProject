"""
Greedy Layer Stacking with Per-Junction Adapters

The key insight: greedy stacking without adapters breaks because each new block
compounds distribution shift downstream. Per-junction adapters absorb each
block's mismatch locally, preventing compound error.

Algorithm:
  Iteration 1: Spectral screen → pick best block A → insert adapter at A's junction
  Iteration 2: Spectral screen modified model → pick best block B → insert adapter at B's junction
  ...each adapter is zero-init (starts as identity), optionally trained

Adapter training options:
  1. No training (identity) — adapter just provides a learnable correction point
  2. Task-utility training — train adapter to maximize math probe score (not KL with baseline)
  3. Minimal KL training — few steps, low LR, just smooth the junction

Based on GPT-5.4 Pro's recommendation: zero-init residual adapters, train against
task utility NOT baseline KL, freeze adapters before adding next block.
"""

import sys, os, json, time, copy, torch, gc, argparse
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', 'scripts'))

from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe

RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/greedy_stacking_adapters")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Junction Adapter (from V4, with improvements)
# =============================================================================

class JunctionAdapter(nn.Module):
    """
    Residual bottleneck adapter. Starts as identity (near-zero up-projection).
    Unlike V4 which used KL loss, this version is trained against task utility.
    """
    def __init__(self, hidden_dim, bottleneck_dim=256, init_scale=0.001):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        # Even smaller init than V4 — start very close to identity
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=init_scale)

    def forward(self, x):
        # Keep everything in input dtype — adapter weights adapt
        dtype = x.dtype
        d = self.down.weight.to(dtype)
        u = self.up.weight.to(dtype)
        residual = nn.functional.linear(
            nn.functional.gelu(nn.functional.linear(x, d)),
            u
        )
        return x + residual


class AdapterWrappedLayer(nn.Module):
    """Wraps a transformer layer with an adapter AFTER its forward pass."""
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
        else:
            return self.adapter(output)


# =============================================================================
# Spectral Screening (same as greedy_stacking.py)
# =============================================================================

SPECTRAL_PROMPTS = [
    "The theory of general relativity states that",
    "In Python, a decorator is a function that",
    "What is 78313 multiplied by 88537?",
    "A linked list is a data structure where",
]


def _run_layers(inner, h, start, end, pos_embeds):
    for i in range(start, end):
        out = inner.layers[i](h, position_embeddings=pos_embeds, use_cache=False)
        h = out[0] if isinstance(out, tuple) else out
    return h


def spectral_screen(model, tokenizer, step=2, block_sizes=None,
                    exclude_ranges=None):
    """
    Compute displacement_rho for candidate blocks on the current model state.
    Returns sorted list of candidates (best first = lowest displacement_rho).
    """
    if block_sizes is None:
        block_sizes = [1, 2, 3, 5]
    if exclude_ranges is None:
        exclude_ranges = []

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

            # Skip excluded ranges
            skip = False
            for ex_start, ex_end in exclude_ranges:
                if not (end <= ex_start or start >= ex_end):
                    skip = True
                    break
            if skip:
                continue

            # Compute displacement_rho
            displacements = []
            for prompt in SPECTRAL_PROMPTS:
                inp = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=64).to(device)

                with torch.no_grad():
                    # Baseline logits
                    h = inner.embed_tokens(inp["input_ids"])
                    seq_len = h.shape[1]
                    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                    pos_embeds = inner.rotary_emb(h, pos_ids)
                    h_base = _run_layers(inner, h, 0, N, pos_embeds)
                    logits_base = model.lm_head(inner.norm(h_base))

                    # Duplicated: insert [start:end) after end
                    dup_order = list(range(end)) + list(range(start, end)) + list(range(end, N))
                    dup_layers = [original_layers[idx] for idx in dup_order]
                    inner.layers = nn.ModuleList(dup_layers)
                    model.config.num_hidden_layers = len(dup_layers)

                    h = inner.embed_tokens(inp["input_ids"])
                    h_dup = _run_layers(inner, h, 0, len(dup_layers), pos_embeds)
                    logits_dup = model.lm_head(inner.norm(h_dup))

                    # Restore
                    inner.layers = nn.ModuleList(original_layers)
                    model.config.num_hidden_layers = N

                    # Displacement
                    diff = (logits_dup - logits_base).float()
                    base_norm = logits_base.float().norm()
                    disp = diff.norm() / (base_norm + 1e-8)
                    displacements.append(disp.item())

            mean_disp = np.mean(displacements)
            candidates.append({
                "start": start, "end": end,
                "block_size": bs,
                "displacement_rho": mean_disp,
            })

    candidates.sort(key=lambda x: x["displacement_rho"])
    return candidates


# =============================================================================
# Build multi-block model with adapters
# =============================================================================

def build_model_with_adapters(model, blocks_with_adapters, original_layers, original_N):
    """
    Apply multiple blocks with adapters to the model.

    blocks_with_adapters: list of (start, end, adapter_or_None)
    Each adapter is inserted after the EXIT layer of the duplicated block.
    """
    inner = model.model

    # Sort blocks by start position
    sorted_blocks = sorted(blocks_with_adapters, key=lambda x: x[0])

    # Build layer order
    order = []
    adapters_at = {}  # layer_index_in_order -> adapter

    prev_j = 0
    for (i, j, adapter) in sorted_blocks:
        # Original layers from prev_j to j
        order.extend(list(range(prev_j, j)))
        # Duplicate [i, j)
        dup_start_idx = len(order)
        order.extend(list(range(i, j)))
        # Adapter goes after the last layer of the duplicate
        if adapter is not None:
            adapters_at[len(order) - 1] = adapter
        prev_j = j
    order.extend(list(range(prev_j, original_N)))

    # Build new layer list
    new_layers = []
    for idx, layer_idx in enumerate(order):
        layer = original_layers[layer_idx]
        if idx in adapters_at:
            layer = AdapterWrappedLayer(layer, adapters_at[idx])
        new_layers.append(layer)

    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    return order


def restore_model(model, original_layers, original_N):
    inner = model.model
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = original_N


# =============================================================================
# Adapter Training (task-utility, NOT KL)
# =============================================================================

def train_adapter_task_utility(model, tokenizer, adapter, num_steps=50, lr=5e-4):
    """
    Train a single adapter to maximize math probe quality.

    Instead of KL with baseline (which fights improvement), we use
    a proxy: minimize next-token cross-entropy on a small set of
    prompts with known correct continuations. This encourages the
    adapter to make the model produce correct outputs, not baseline outputs.
    """
    device = next(model.parameters()).device

    # Simple training data: prompts with expected continuations
    train_data = [
        ("What is 9999 multiplied by 9999? Answer: ", "99980001"),
        ("What is 2 to the power of 10? Answer: ", "1024"),
        ("The capital of France is ", "Paris"),
        ("What is 7 times 8? Answer: ", "56"),
        ("The chemical symbol for water is ", "H2O"),
        ("What is 100 divided by 4? Answer: ", "25"),
        ("The largest planet in our solar system is ", "Jupiter"),
        ("What is the square root of 144? Answer: ", "12"),
    ]

    # Freeze everything except adapter
    for param in model.parameters():
        param.requires_grad = False
    # Explicitly enable grad on adapter (must iterate adapter directly,
    # not through model.parameters() which may not find wrapped adapters)
    for param in adapter.parameters():
        param.requires_grad = True

    # Verify adapter params are trainable
    trainable = sum(p.requires_grad for p in adapter.parameters())
    total_adapter = sum(1 for _ in adapter.parameters())
    assert trainable == total_adapter, f"Only {trainable}/{total_adapter} adapter params are trainable"

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.01)

    adapter_params = sum(p.numel() for p in adapter.parameters())
    print(f"    Training adapter ({adapter_params:,} params) for {num_steps} steps...")

    model.train()
    for step in range(num_steps):
        total_loss = 0
        optimizer.zero_grad()

        for prompt, target in train_data:
            full = prompt + target
            inp = tokenizer(full, return_tensors="pt", truncation=True,
                           max_length=64).to(device)
            prompt_len = len(tokenizer(prompt)["input_ids"])

            out = model(**inp, use_cache=False)
            logits = out.logits[:, prompt_len-1:-1, :]
            targets = inp["input_ids"][:, prompt_len:]

            loss = nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            loss.backward()
            total_loss += loss.item()

        nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 25 == 0:
            print(f"      Step {step+1}/{num_steps} loss={total_loss/len(train_data):.4f}")

    model.eval()

    # Re-freeze adapter for future use
    for param in adapter.parameters():
        param.requires_grad = False

    return adapter


# =============================================================================
# Main Greedy Stacking Loop with Adapters
# =============================================================================

def run_greedy_stacking_adapters(
    model_path,
    max_iterations=5,
    top_k=5,
    step=2,
    block_sizes=None,
    adapter_bottleneck=256,
    adapter_train_steps=50,
    train_adapters=True,
    force=False,
):
    if block_sizes is None:
        block_sizes = [1, 2, 3, 5]

    output_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Greedy Stacking with Per-Junction Adapters")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Max iterations: {max_iterations}")
    print(f"Adapter bottleneck: {adapter_bottleneck}")
    print(f"Train adapters: {train_adapters} ({adapter_train_steps} steps)")
    print(f"Force continue: {force}")

    model, tokenizer = load_original_model(model_path)
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    inner = model.model
    original_layers = list(inner.layers)
    original_N = len(original_layers)

    # Baseline
    print(f"\n--- Baseline ---")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    baseline = run_math_probe(gen_fn, verbose=False)
    baseline_score = baseline["score"]
    print(f"Baseline: {baseline_score:.4f}")

    blocks_with_adapters = []  # list of (start, end, adapter)
    scores = [baseline_score]
    iteration_results = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}")

        # Apply current blocks+adapters to model for spectral screening
        if blocks_with_adapters:
            build_model_with_adapters(model, blocks_with_adapters,
                                     original_layers, original_N)

        # Spectral screen — always in ORIGINAL index space
        # Restore to original model for screening
        restore_model(model, original_layers, original_N)

        exclude = [(b[0], b[1]) for b in blocks_with_adapters]
        print(f"\n--- Spectral Screening ({original_N} layers, excluding {exclude}) ---")

        candidates = spectral_screen(model, tokenizer, step=step,
                                     block_sizes=block_sizes,
                                     exclude_ranges=exclude)

        # Restore original for clean evaluation
        restore_model(model, original_layers, original_N)

        print(f"  Screened {len(candidates)} candidates")
        for c in candidates[:5]:
            print(f"    ({c['start']},{c['end']}) disp_rho={c['displacement_rho']:.4f}")

        # Evaluate top-K
        eval_k = min(top_k, len(candidates))
        print(f"\n--- Evaluating top-{eval_k} with adapters ---")

        best_score = -float('inf')
        best_block = None
        best_adapter = None
        eval_results = []

        for rank, cand in enumerate(candidates[:eval_k]):
            block = (cand["start"], cand["end"])

            # Create adapter in float32 for stable training (small params, no VRAM issue)
            adapter = JunctionAdapter(hidden_dim, adapter_bottleneck).to(device).float()

            # Build model with all existing blocks + this candidate + adapter
            test_blocks = blocks_with_adapters + [(block[0], block[1], adapter)]
            build_model_with_adapters(model, test_blocks, original_layers, original_N)

            # Optionally train the adapter
            if train_adapters and adapter_train_steps > 0:
                train_adapter_task_utility(
                    model, tokenizer, adapter,
                    num_steps=adapter_train_steps
                )

            # Evaluate
            model.eval()
            result = run_math_probe(gen_fn, verbose=False)
            score = result["score"]
            delta = score - scores[-1]

            eval_results.append({
                "block": list(block),
                "score": score,
                "delta": delta,
                "displacement_rho": cand["displacement_rho"],
            })

            indicator = " ***BEST***" if score > best_score else ""
            print(f"  [{rank+1}/{eval_k}] ({block[0]},{block[1]}) "
                  f"score={score:.4f} delta={delta:+.4f}{indicator}")

            if score > best_score:
                best_score = score
                best_block = block
                # Deep copy the trained adapter weights
                best_adapter = JunctionAdapter(hidden_dim, adapter_bottleneck).to(device).to(torch.bfloat16)
                best_adapter.load_state_dict(adapter.state_dict())

            # Restore for next candidate
            restore_model(model, original_layers, original_N)

        # Record
        improved = best_score > scores[-1]
        iter_result = {
            "iteration": iteration,
            "evaluations": eval_results,
            "best_block": list(best_block) if best_block else None,
            "best_score": best_score,
            "improved": improved,
            "delta": best_score - scores[-1],
        }
        iteration_results.append(iter_result)

        # Termination check
        if best_block is None:
            print(f"\n--- STOPPING: No candidates ---")
            break

        if not improved and not force:
            print(f"\n--- STOPPING: No improvement ---")
            print(f"  Previous: {scores[-1]:.4f}, Best candidate: {best_score:.4f}")
            break

        # Accept
        blocks_with_adapters.append((best_block[0], best_block[1], best_adapter))
        scores.append(best_score)

        status = "IMPROVED" if improved else "DECLINED (forced)"
        print(f"\n--- {status}: ({best_block[0]},{best_block[1]}) + adapter ---")
        print(f"  Score: {scores[-2]:.4f} -> {best_score:.4f} ({best_score - scores[-2]:+.4f})")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Baseline: {baseline_score:.4f}")
    peak_score = baseline_score
    for i, ((s, e, _), score) in enumerate(zip(blocks_with_adapters, scores[1:])):
        delta = score - (scores[i] if i < len(scores)-1 else baseline_score)
        marker = " <-- PEAK" if score > peak_score else ""
        if score > peak_score:
            peak_score = score
        print(f"  Iter {i+1}: ({s},{e}) + adapter -> {score:.4f} ({delta:+.4f}){marker}")
    print(f"\nFinal: {scores[-1]:.4f} (delta from baseline: {scores[-1]-baseline_score:+.4f})")
    print(f"Blocks: {[(b[0], b[1]) for b in blocks_with_adapters]}")

    # Save
    save_data = {
        "experiment": "greedy_stacking_adapters",
        "timestamp": datetime.now().isoformat(),
        "baseline": baseline_score,
        "scores": scores,
        "blocks": [(b[0], b[1]) for b in blocks_with_adapters],
        "iterations": iteration_results,
        "adapter_bottleneck": adapter_bottleneck,
        "adapter_train_steps": adapter_train_steps,
        "train_adapters": train_adapters,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Save adapter weights
    for i, (s, e, adapter) in enumerate(blocks_with_adapters):
        torch.save(adapter.state_dict(),
                   output_dir / f"adapter_iter{i+1}_{s}_{e}.pt")

    print(f"\nSaved to {output_dir}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return save_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--block-sizes", type=str, default="1,2,3,5")
    parser.add_argument("--adapter-bottleneck", type=int, default=256)
    parser.add_argument("--adapter-steps", type=int, default=50)
    parser.add_argument("--no-train-adapters", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    block_sizes = [int(x) for x in args.block_sizes.split(",")]

    run_greedy_stacking_adapters(
        model_path=args.model,
        max_iterations=args.max_iterations,
        top_k=args.top_k,
        step=args.step,
        block_sizes=block_sizes,
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_train_steps=args.adapter_steps,
        train_adapters=not args.no_train_adapters,
        force=args.force,
    )


if __name__ == "__main__":
    main()
