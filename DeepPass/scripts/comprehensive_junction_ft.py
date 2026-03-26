"""
Comprehensive Junction Fine-Tuning Experiments

Tests junction FT on:
1. 7B with GOOD config (10,11) — best brain scanner result
2. 7B with BAD config (15,18) — already tested, for comparison
3. 7B with GOOD config (18,21) — second-best config
4. 72B with Ng's config (45,52) — reproduction
5. 72B with OUR config (50,60) — the one that beats Ng

For each: measure pre-FT score, run junction FT, measure post-FT score.
"""

import sys, os, json, copy, time, torch, gc
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe

RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/junction_ft_comprehensive")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_junction_ft_experiment(model_path, i, j, num_steps=100, lr=5e-6, tag=""):
    """Run a single junction FT experiment."""
    print(f"\n{'='*60}")
    print(f"JUNCTION FT: {tag} — config ({i},{j})")
    print(f"{'='*60}")

    model, tokenizer = load_original_model(model_path)
    inner = model.model
    layers = list(inner.layers)
    N = len(layers)

    # Baseline (no duplication)
    print("\n--- Baseline (no duplication) ---")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    baseline = run_math_probe(gen_fn, verbose=False)
    print(f"  Baseline: {baseline['score']:.4f}")

    # Build duplicated model with deep copies at junction
    print(f"\n--- Building duplicated model ({i},{j}) ---")
    new_layers = layers[:j]
    for idx in range(i, j):
        new_layers.append(copy.deepcopy(layers[idx]))
    new_layers.extend(layers[j:])

    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    new_N = len(new_layers)
    print(f"  {N} → {new_N} layers")

    # Pre-FT score
    print("\n--- Pre-FT math probe ---")
    pre = run_math_probe(gen_fn, verbose=False)
    pre_delta = pre['score'] - baseline['score']
    print(f"  Pre-FT: {pre['score']:.4f} (delta from baseline: {pre_delta:+.4f})")

    # Junction layer indices
    junction = [j - 1, j]  # end of first pass, start of second pass
    print(f"  Junction layers: {junction}")

    # Freeze everything except junction
    for param in model.parameters():
        param.requires_grad = False
    trainable = 0
    for idx in junction:
        if idx < new_N:
            for param in inner.layers[idx].parameters():
                param.requires_grad = True
                trainable += param.numel()
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    # Training prompts — mix of reasoning and factual
    prompts = [
        "The theory of general relativity states that",
        "In Python, a decorator is a function that",
        "To solve a quadratic equation, you can use",
        "Machine learning models are trained by",
        "The derivative of sin(x) is",
        "What is 78313 multiplied by 88537?",
        "The cube root of 74088 is approximately",
        "What is 9999 multiplied by 9999?",
        "The square root of 152399025 is",
        "What is 123456789 multiplied by 987654321?",
        "A linked list is a data structure where",
        "The speed of light in a vacuum is approximately",
    ]

    # Self-distillation targets
    print("\n--- Generating targets ---")
    model.eval()
    targets = []
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt", truncation=True,
                       max_length=64).to(model.device)
        with torch.no_grad():
            out = model(**inp, use_cache=False)
        targets.append((inp, out.logits.detach()))

    # Fine-tune
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )

    print(f"\n--- Fine-tuning {num_steps} steps ---")
    model.train()
    losses = []
    for step in range(num_steps):
        total_loss = 0
        for inp, target_logits in targets:
            out = model(**inp, use_cache=False)
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out.logits[:, :-1, :], dim=-1),
                nn.functional.softmax(target_logits[:, :-1, :], dim=-1),
                reduction='batchmean'
            )
            loss.backward()
            total_loss += loss.item()
        avg = total_loss / len(targets)
        losses.append(avg)
        optimizer.step()
        optimizer.zero_grad()
        if (step + 1) % 25 == 0:
            print(f"    Step {step+1}/{num_steps} loss={avg:.6f}")

    # Post-FT score
    print("\n--- Post-FT math probe ---")
    model.eval()
    post = run_math_probe(gen_fn, verbose=False)
    post_delta = post['score'] - baseline['score']
    ft_improvement = post['score'] - pre['score']

    print(f"\n  Baseline:  {baseline['score']:.4f}")
    print(f"  Pre-FT:    {pre['score']:.4f} (delta: {pre_delta:+.4f})")
    print(f"  Post-FT:   {post['score']:.4f} (delta: {post_delta:+.4f})")
    print(f"  FT gain:   {ft_improvement:+.4f}")
    print(f"  FT recovered: {ft_improvement/abs(pre_delta)*100:.1f}% of {'loss' if pre_delta < 0 else 'gain'}" if abs(pre_delta) > 0.001 else "")

    result = {
        "tag": tag,
        "model": model_path,
        "config": [i, j],
        "num_layers_original": N,
        "num_layers_duplicated": new_N,
        "baseline_score": baseline['score'],
        "pre_ft_score": pre['score'],
        "post_ft_score": post['score'],
        "pre_ft_delta": pre_delta,
        "post_ft_delta": post_delta,
        "ft_improvement": ft_improvement,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total,
        "num_steps": num_steps,
        "lr": lr,
        "final_loss": losses[-1] if losses else 0,
    }

    # Cleanup
    del model, tokenizer, targets
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    all_results = []

    # 7B experiments
    M7B = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"

    # Good config
    r = run_junction_ft_experiment(M7B, 10, 11, num_steps=100, tag="7B_good_10_11")
    all_results.append(r)

    # Second good config
    r = run_junction_ft_experiment(M7B, 18, 21, num_steps=100, tag="7B_good_18_21")
    all_results.append(r)

    # Bad config (for comparison)
    r = run_junction_ft_experiment(M7B, 4, 9, num_steps=100, tag="7B_bad_4_9")
    all_results.append(r)

    # 72B experiments (solo GPU — nothing else should be running)
    M72B = "/blue/cis4914/jietao/DeepPass/models/full/calme-2.1-qwen2-72b"

    # Ng's config
    r = run_junction_ft_experiment(M72B, 45, 52, num_steps=50, lr=5e-6, tag="72B_ng_45_52")
    all_results.append(r)

    # Our best config
    r = run_junction_ft_experiment(M72B, 50, 60, num_steps=50, lr=5e-6, tag="72B_ours_50_60")
    all_results.append(r)

    # Summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE JUNCTION FT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Tag':>20} {'Config':>10} {'Baseline':>10} {'Pre-FT':>10} {'Post-FT':>10} {'FT Gain':>10} {'Params%':>8}")
    for r in all_results:
        print(f"{r['tag']:>20} ({r['config'][0]},{r['config'][1]})"
              f" {r['baseline_score']:10.4f}"
              f" {r['pre_ft_score']:10.4f}"
              f" {r['post_ft_score']:10.4f}"
              f" {r['ft_improvement']:+10.4f}"
              f" {r['trainable_pct']:7.4f}%")

    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
