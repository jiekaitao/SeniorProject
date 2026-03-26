"""
DeepPass Spectral Analysis

Replace Ng's brute-force sweep with Jacobian spectral analysis.
Computes displacement rho, perturbation rho, and fixed-point distance
for each layer block to predict which blocks benefit from duplication.
"""

import json, os, sys, time, torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import load_original_model, generate_no_cache
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_hidden_states_at_layer(model, tokenizer, prompt, target_layer):
    """Get hidden states at a specific layer using a forward hook."""
    inner = model.model if hasattr(model, 'model') else model.transformer
    layers = inner.layers if hasattr(inner, 'layers') else inner.h

    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['hidden'] = output[0].detach().clone()
        else:
            captured['hidden'] = output.detach().clone()

    # Also capture the input to the target layer
    def input_hook_fn(module, input, output):
        if isinstance(input, tuple):
            captured['input'] = input[0].detach().clone()
        else:
            captured['input'] = input.detach().clone()

    # Register hooks
    hooks = []
    if target_layer > 0:
        hooks.append(layers[target_layer - 1].register_forward_hook(hook_fn))
    hooks.append(layers[target_layer].register_forward_hook(
        lambda m, i, o: captured.update({'layer_input': i[0].detach().clone() if isinstance(i, tuple) else i.detach().clone()})
    ))

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        model(**inputs, use_cache=False)

    for h in hooks:
        h.remove()

    return captured


def compute_block_metrics_v2(
    model, tokenizer, prompts, start_layer, end_layer, eps=1e-3
):
    """
    Compute spectral metrics for block [start, end) using full model forward passes.

    Strategy: Run the full model normally to get baseline hidden states,
    then compare against a model with the block duplicated. This avoids
    needing to run individual layers with the correct mask/position setup.
    """
    inner = model.model if hasattr(model, 'model') else model.transformer
    attr_name = 'layers' if hasattr(inner, 'layers') else 'h'
    original_layers = list(getattr(inner, attr_name))
    N = len(original_layers)

    device = next(model.parameters()).device

    disp_rhos = []
    residuals = []
    pert_rhos = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=128).to(device)

        with torch.no_grad():
            # 1. Normal forward pass — capture hidden states after end_layer
            hidden_normal = {}
            def hook_after_end(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                hidden_normal['h'] = h.detach().clone()

            hook1 = original_layers[min(end_layer, N-1)].register_forward_hook(hook_after_end)
            model(**inputs, use_cache=False)
            hook1.remove()
            h_normal = hidden_normal.get('h')
            if h_normal is None:
                continue

            # 2. Duplicated forward pass — run block [start, end) twice
            layer_order = list(range(end_layer)) + list(range(start_layer, end_layer)) + list(range(end_layer, N))
            dup_layers = [original_layers[k] for k in layer_order]
            setattr(inner, attr_name, nn.ModuleList(dup_layers))
            model.config.num_hidden_layers = len(dup_layers)

            hidden_dup = {}
            # The duplicated block ends at position end_layer + (end_layer - start_layer)
            dup_end_idx = end_layer + (end_layer - start_layer)
            if dup_end_idx < len(dup_layers):
                hook2 = dup_layers[dup_end_idx].register_forward_hook(
                    lambda m, i, o: hidden_dup.update({'h': (o[0] if isinstance(o, tuple) else o).detach().clone()})
                )
            else:
                hook2 = dup_layers[-1].register_forward_hook(
                    lambda m, i, o: hidden_dup.update({'h': (o[0] if isinstance(o, tuple) else o).detach().clone()})
                )

            model(**inputs, use_cache=False)
            hook2.remove()

            # Restore original
            setattr(inner, attr_name, nn.ModuleList(original_layers))
            model.config.num_hidden_layers = N

            h_dup = hidden_dup.get('h')
            if h_dup is None:
                continue

            # 3. Compute displacement metrics
            # diff1 = how much the block changes the hidden state (1st pass effect)
            # We approximate this by comparing final logits or last hidden states
            # Actually, let's use the output logits directly

            # Run normal model to get logits
            out_normal = model(**inputs, use_cache=False)
            logits_normal = out_normal.logits.detach()

            # Run duplicated model to get logits
            dup_layers2 = [original_layers[k] for k in layer_order]
            setattr(inner, attr_name, nn.ModuleList(dup_layers2))
            model.config.num_hidden_layers = len(dup_layers2)
            out_dup = model(**inputs, use_cache=False)
            logits_dup = out_dup.logits.detach()

            # Restore
            setattr(inner, attr_name, nn.ModuleList(original_layers))
            model.config.num_hidden_layers = N

            # 4. Metrics from logit differences
            # "displacement" = how much duplication changes the output
            logit_diff = (logits_dup - logits_normal).float()
            logit_norm = logits_normal.float().norm(dim=-1).mean().item()
            diff_norm = logit_diff.norm(dim=-1).mean().item()

            if logit_norm > 1e-10:
                disp_rhos.append(diff_norm / logit_norm)
            residuals.append(diff_norm)

            # 5. Perturbation: add noise to embeddings, measure output change
            # This estimates sensitivity of the block
            if hasattr(inner, 'embed_tokens'):
                emb = inner.embed_tokens(inputs['input_ids'])
            elif hasattr(inner, 'wte'):
                emb = inner.wte(inputs['input_ids'])
            else:
                continue

            noise = torch.randn_like(emb) * eps
            # Can't easily perturb mid-model, so we measure full-model sensitivity
            # as a proxy for block sensitivity
            out_pert = model(inputs_embeds=emb + noise, use_cache=False)
            logits_pert = out_pert.logits.detach()
            pert_diff = (logits_pert - logits_normal).float().norm(dim=-1).mean().item()
            pert_input = noise.float().norm(dim=-1).mean().item()
            if pert_input > 1e-10:
                pert_rhos.append(pert_diff / pert_input)

    return {
        "displacement_rho": float(np.mean(disp_rhos)) if disp_rhos else 0,
        "residual": float(np.mean(residuals)) if residuals else 0,
        "perturbation_rho": float(np.mean(pert_rhos)) if pert_rhos else 0,
        "displacement_rho_std": float(np.std(disp_rhos)) if disp_rhos else 0,
    }


def spectral_sweep(
    model_path: str,
    output_dir: str,
    step: int = 1,
    block_sizes: List[int] = None,
    num_prompts: int = 8,
    device_map: str = "auto",
):
    """Run spectral analysis sweep over all layer blocks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    inner = model.model if hasattr(model, 'model') else model.transformer
    layers = inner.layers if hasattr(inner, 'layers') else inner.h
    N = len(list(layers))
    print(f"Model has {N} layers")

    if block_sizes is None:
        block_sizes = list(range(1, min(N // 2, 15) + 1))

    prompts = [
        "The theory of general relativity describes",
        "In mathematics, a topological space is",
        "def fibonacci(n):\n    if n <= 1:",
        "What is 123 multiplied by 456?",
        "The mitochondria is the powerhouse of",
        "According to quantum mechanics, particles",
        "Once upon a time in a land far away",
        "A linked list is a data structure where",
    ][:num_prompts]

    results = {}
    configs = []
    for bs in block_sizes:
        for start in range(0, N - bs + 1, step):
            configs.append((start, start + bs))

    total = len(configs)
    print(f"\nSweeping {total} configs ({len(block_sizes)} block sizes)")
    t_start = time.time()

    for idx, (start, end) in enumerate(configs):
        t0 = time.time()
        try:
            metrics = compute_block_metrics_v2(
                model, tokenizer, prompts, start, end
            )
            results[(start, end)] = metrics
            elapsed = time.time() - t0
            print(f"  [{idx+1:4d}/{total}] ({start:3d},{end:3d}) bs={end-start:2d} "
                  f"disp_rho={metrics['displacement_rho']:.4f} "
                  f"resid={metrics['residual']:.6f} "
                  f"pert_rho={metrics['perturbation_rho']:.4f} "
                  f"({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [{idx+1:4d}/{total}] ({start:3d},{end:3d}) ERROR: {e}")
            results[(start, end)] = {"error": str(e)}

        if (idx + 1) % 20 == 0:
            save_results(results, N, output_dir)

    total_time = time.time() - t_start
    print(f"\nSweep done in {total_time:.1f}s")
    save_results(results, N, output_dir)
    generate_heatmaps(results, N, output_dir)
    predict_best(results, N, output_dir)
    return results


def save_results(results, N, output_dir):
    output_dir = Path(output_dir)
    ser = {f"{i},{j}": v for (i, j), v in results.items()}
    with open(output_dir / "spectral_results.json", "w") as f:
        json.dump({"num_layers": N, "results": ser}, f, indent=2)


def predict_best(results, N, output_dir):
    output_dir = Path(output_dir)
    scored = []
    for (i, j), m in results.items():
        if "error" in m:
            continue
        bs = j - i
        disp = m["displacement_rho"]
        resid = m["residual"]

        # Score: low displacement rho + high residual + middle position + good size
        center = (i + j) / 2
        edge_dist = min(center, N - center)
        pos_score = min(1.0, edge_dist / (N * 0.15))
        size_score = np.exp(-((bs - 7) / 5) ** 2)
        resid_score = np.tanh(resid * 10)

        # Lower displacement = block output is more similar with/without duplication
        # This means the block is already near convergence
        # We want moderate displacement (block does something, but not too much)
        disp_score = np.exp(-((disp - 0.05) / 0.1) ** 2)

        score = disp_score * 0.35 + resid_score * 0.30 + pos_score * 0.20 + size_score * 0.15
        scored.append(((i, j), score, m))

    scored.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print(f"TOP 10 PREDICTED CONFIGS")
    print(f"{'='*60}")
    for rank, ((i, j), score, m) in enumerate(scored[:10], 1):
        print(f"  {rank:2d}. ({i:3d},{j:3d}) score={score:.4f} "
              f"disp={m['displacement_rho']:.4f} resid={m['residual']:.6f}")

    preds = {"top_10": [{"config": [i, j], "score": s} for (i, j), s, _ in scored[:10]]}
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(preds, f, indent=2, default=str)


def generate_heatmaps(results, N, output_dir):
    output_dir = Path(output_dir)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for metric_key, title in [
        ("displacement_rho", "Displacement ρ"),
        ("residual", "Residual ||F(F(h))-F(h)||"),
        ("perturbation_rho", "Perturbation ρ"),
    ]:
        heatmap = np.full((N, N + 1), np.nan)
        for (i, j), v in results.items():
            if "error" not in v and metric_key in v:
                heatmap[i, j] = v[metric_key]

        valid = heatmap[~np.isnan(heatmap)]
        if len(valid) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap, cmap='viridis', origin='upper', aspect='auto')
        ax.set_xlabel('End layer j')
        ax.set_ylabel('Start layer i')
        ax.set_title(f'{title} — N={N}')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_{metric_key}.png", dpi=150)
        plt.close()

    print(f"Heatmaps saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--block-sizes", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=8)
    args = parser.parse_args()

    if args.output is None:
        name = Path(args.model).name
        args.output = f"/blue/cis4914/jietao/DeepPass/results/spectral_{name}"

    block_sizes = None
    if args.block_sizes:
        block_sizes = [int(x) for x in args.block_sizes.split(",")]

    spectral_sweep(args.model, args.output, step=args.step,
                   block_sizes=block_sizes, num_prompts=args.num_prompts)
