"""
Oracle Seam Patching V2 — Fixed implementation.

V1 bug: hooks on shared layer modules fire on both passes since they're the
same Python object. Fix: manually run the forward pass layer-by-layer,
intercepting hidden states at the seam.

For each duplicated block, tests:
    h_patched = h1 + alpha * (h2 - h1)
where h1 = after first pass, h2 = after second pass.
"""
import sys, os, json, torch, torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from layer_duplicator import load_original_model
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe


def build_layer_order(blocks, N):
    sorted_blocks = sorted(blocks)
    order = []
    prev = 0
    for (i, j) in sorted_blocks:
        order.extend(list(range(prev, j)))
        order.extend(list(range(i, j)))
        prev = j
    order.extend(list(range(prev, N)))
    return order


def find_seam_positions(layer_order, block_start, block_end):
    """
    Find step indices in layer_order where first and second pass of a block end.
    Returns (first_pass_end_step, second_pass_end_step).
    """
    last_layer = block_end - 1
    occurrences = [step for step, layer_idx in enumerate(layer_order) if layer_idx == last_layer]
    if len(occurrences) < 2:
        raise ValueError(f"Block ({block_start},{block_end}) not found duplicated. "
                         f"Layer {last_layer} appears {len(occurrences)} time(s)")
    return occurrences[0], occurrences[1]


def generate_with_seam_patch(model, tokenizer, prompt, layer_order, original_layers,
                              patch_block, alpha, max_new_tokens=64):
    """
    Generate text token-by-token with manual layer-by-layer forward pass.
    At the seam of patch_block, applies: h = h1 + alpha * (h2 - h1).

    If patch_block is None or alpha is 1.0, runs normally (no patching).
    """
    inner = model.model
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # Find seam positions if patching
    first_end = second_end = None
    if patch_block is not None and alpha != 1.0:
        first_end, second_end = find_seam_positions(layer_order, patch_block[0], patch_block[1])

    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Embedding
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            # Layer-by-layer forward
            h_after_first_pass = None
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

                # Cache h after first pass ends
                if step_idx == first_end:
                    h_after_first_pass = h.clone()

                # Apply alpha patch after second pass ends
                if step_idx == second_end and h_after_first_pass is not None:
                    h = h_after_first_pass + alpha * (h - h_after_first_pass)

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


def evaluate_config(model, tokenizer, original_layers, layer_order,
                    patch_block, alpha, name):
    """Run dual probe with seam patching."""

    def gen(p):
        return generate_with_seam_patch(
            model, tokenizer, p, layer_order, original_layers,
            patch_block, alpha, max_new_tokens=64)

    def gen_long(p):
        return generate_with_seam_patch(
            model, tokenizer, p, layer_order, original_layers,
            patch_block, alpha, max_new_tokens=128)

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5

    print(f"    {name:35s}: math={math_r['score']:.4f} eq={eq_r['score']:.1f} combined={combined:.2f}")
    return {
        'name': name,
        'patch_block': list(patch_block) if patch_block else None,
        'alpha': alpha,
        'math': math_r['score'],
        'eq': eq_r['score'],
        'combined': combined,
    }


def main():
    print("=" * 60)
    print("ORACLE SEAM PATCHING V2 — 72B")
    print("=" * 60)

    model, tokenizer = load_original_model('models/full/calme-2.1-qwen2-72b')
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)
    print(f"Loaded: {N} layers\n")

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_results = {}

    # ---- Test 1: Single block (45,52) — Ng's config ----
    print("TEST 1: Single block (45,52)")
    blocks = [(45, 52)]
    order = build_layer_order(blocks, N)
    results = []
    for alpha in alphas:
        name = f"(45,52) alpha={alpha:.2f}"
        patch = None if alpha == 1.0 else (45, 52)
        r = evaluate_config(model, tokenizer, original_layers, order, patch, alpha, name)
        results.append(r)
    all_results['single_45_52'] = results

    # ---- Test 2: Single block (50,60) — our best single ----
    print("\nTEST 2: Single block (50,60)")
    blocks = [(50, 60)]
    order = build_layer_order(blocks, N)
    results = []
    for alpha in alphas:
        name = f"(50,60) alpha={alpha:.2f}"
        patch = None if alpha == 1.0 else (50, 60)
        r = evaluate_config(model, tokenizer, original_layers, order, patch, alpha, name)
        results.append(r)
    all_results['single_50_60'] = results

    # ---- Test 3: Best pair (0,7)+(45,52), patch each block ----
    print("\nTEST 3: Pair (0,7)+(45,52), patching (0,7)")
    blocks = [(0, 7), (45, 52)]
    order = build_layer_order(blocks, N)
    results = []
    # Alpha=1.0 baseline (no patch)
    r = evaluate_config(model, tokenizer, original_layers, order, None, 1.0, "no_patch alpha=1.00")
    results.append(r)
    # Patch (0,7) at different alphas
    for alpha in [0.0, 0.25, 0.5, 0.75]:
        name = f"patch(0,7) alpha={alpha:.2f}"
        r = evaluate_config(model, tokenizer, original_layers, order, (0, 7), alpha, name)
        results.append(r)
    # Patch (45,52) at different alphas
    for alpha in [0.0, 0.25, 0.5, 0.75]:
        name = f"patch(45,52) alpha={alpha:.2f}"
        r = evaluate_config(model, tokenizer, original_layers, order, (45, 52), alpha, name)
        results.append(r)
    all_results['pair_0_7_45_52'] = results

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY — Optimal Alpha per Config")
    print("=" * 60)
    for config_name, results in all_results.items():
        best = max(results, key=lambda x: x['combined'])
        print(f"\n{config_name}: best alpha={best['alpha']:.2f} combined={best['combined']:.2f}")
        for r in results:
            marker = " <-- BEST" if r['combined'] == best['combined'] else ""
            print(f"  {r['name']:35s}: combined={r['combined']:.2f}{marker}")

    # Save
    os.makedirs('results/data/72b/oracle_seam_patching', exist_ok=True)
    with open('results/data/72b/oracle_seam_patching/v2_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/data/72b/oracle_seam_patching/v2_results.json")


if __name__ == '__main__':
    main()
