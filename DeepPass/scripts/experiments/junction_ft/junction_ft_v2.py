"""
Junction Fine-Tuning V2 — Distill from Original Model

Key improvement: Instead of self-distillation (model distills from itself),
we distill from the ORIGINAL unduplicated model. This teaches the junction
layers to preserve the original model's output quality while gaining the
benefit of extra reasoning depth.

The idea: The original 80-layer model produces good outputs. The 87-layer
duplicated model produces slightly different outputs because of the
distributional shift at the junction. We train the junction layers to
minimize this difference, so the duplicated model acts like a "refined"
version of the original rather than a corrupted one.
"""

import sys, os, json, copy, time, torch, gc
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe

RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/junction_ft_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_junction_ft_v2(model_path, i, j, num_steps=100, lr=5e-6, tag=""):
    """
    Junction FT with distillation from the original unduplicated model.

    1. Load model, get original logits on training prompts (TEACHER)
    2. Build duplicated model (STUDENT)
    3. Freeze everything except junction layers
    4. Train junction layers to match original model's logits
    """
    print(f"\n{'='*60}")
    print(f"JUNCTION FT V2: {tag} — config ({i},{j})")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_original_model(model_path)
    inner = model.model
    layers = list(inner.layers)
    N = len(layers)

    # Training prompts — diverse mix
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
        "The Pythagorean theorem states that in a right triangle",
        "To implement quicksort, you first choose a pivot",
        "The integral of e^x dx equals",
        "In economics, inflation is defined as",
    ]

    # Step 1: Get ORIGINAL model logits (teacher signal)
    print("\n--- Step 1: Collecting teacher logits from original model ---")
    model.eval()
    teacher_data = []
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt", truncation=True,
                       max_length=64).to(model.device)
        with torch.no_grad():
            out = model(**inp, use_cache=False)
        teacher_data.append((inp, out.logits.detach().clone()))

    # Step 2: Baseline score (original model)
    print("\n--- Step 2: Baseline score ---")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    baseline = run_math_probe(gen_fn, verbose=False)
    print(f"  Original model: {baseline['score']:.4f}")

    # Step 3: Build duplicated model with deep copies
    print(f"\n--- Step 3: Building duplicated model ({i},{j}) ---")
    new_layers = layers[:j]
    for idx in range(i, j):
        new_layers.append(copy.deepcopy(layers[idx]))
    new_layers.extend(layers[j:])

    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    new_N = len(new_layers)
    dup_count = j - i
    print(f"  {N} → {new_N} layers (+{dup_count} duplicated)")

    # Step 4: Pre-FT score (duplicated, before training)
    print("\n--- Step 4: Pre-FT score ---")
    pre = run_math_probe(gen_fn, verbose=False)
    pre_delta = pre['score'] - baseline['score']
    print(f"  Pre-FT: {pre['score']:.4f} (delta: {pre_delta:+.4f})")

    # Step 5: Freeze everything except junction layers
    junction = [j - 1, j]  # seam layers
    for param in model.parameters():
        param.requires_grad = False
    trainable = 0
    for idx in junction:
        if idx < new_N:
            for param in inner.layers[idx].parameters():
                param.requires_grad = True
                trainable += param.numel()
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    print(f"  Junction layers: {junction}")
    print(f"  Trainable: {trainable:,} ({pct:.4f}%)")

    # Step 6: Train junction to match ORIGINAL model's logits
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )

    print(f"\n--- Step 6: Training {num_steps} steps (distill from original) ---")
    model.train()
    losses = []
    best_loss = float('inf')

    for step in range(num_steps):
        total_loss = 0
        for inp, teacher_logits in teacher_data:
            student_out = model(**inp, use_cache=False)

            # KL(student || teacher) — train student to match teacher
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(student_out.logits[:, :-1, :] / 2.0, dim=-1),
                nn.functional.softmax(teacher_logits[:, :-1, :] / 2.0, dim=-1),
                reduction='batchmean'
            ) * (2.0 ** 2)  # temperature scaling

            loss.backward()
            total_loss += loss.item()

        avg = total_loss / len(teacher_data)
        losses.append(avg)
        if avg < best_loss:
            best_loss = avg

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 25 == 0:
            print(f"    Step {step+1}/{num_steps} loss={avg:.6f} (best={best_loss:.6f})")

    # Step 7: Post-FT score
    print("\n--- Step 7: Post-FT score ---")
    model.eval()
    post = run_math_probe(gen_fn, verbose=False)
    post_delta = post['score'] - baseline['score']
    ft_gain = post['score'] - pre['score']

    print(f"\n  {'='*50}")
    print(f"  RESULTS: {tag}")
    print(f"  {'='*50}")
    print(f"  Original (no dup): {baseline['score']:.4f}")
    print(f"  Duplicated pre-FT: {pre['score']:.4f} ({pre_delta:+.4f})")
    print(f"  Duplicated post-FT: {post['score']:.4f} ({post_delta:+.4f})")
    print(f"  FT improvement:    {ft_gain:+.4f}")
    if pre_delta > 0:
        print(f"  Post-FT vs original: {post_delta:+.4f} ({'BETTER' if post_delta > pre_delta else 'worse'} than pre-FT)")
    else:
        recovered = ft_gain / abs(pre_delta) * 100 if abs(pre_delta) > 0.001 else 0
        print(f"  Recovered {recovered:.1f}% of duplication damage")

    result = {
        "tag": tag, "config": [i, j],
        "model": model_path,
        "baseline": baseline['score'],
        "pre_ft": pre['score'], "pre_delta": pre_delta,
        "post_ft": post['score'], "post_delta": post_delta,
        "ft_gain": ft_gain,
        "trainable": trainable, "trainable_pct": pct,
        "steps": num_steps, "lr": lr,
        "best_loss": best_loss, "final_loss": losses[-1],
    }

    del model, tokenizer, teacher_data
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    results = []
    M7B = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"
    M72B = "/blue/cis4914/jietao/DeepPass/models/full/calme-2.1-qwen2-72b"

    # 7B: good config
    results.append(run_junction_ft_v2(M7B, 10, 11, num_steps=150, lr=1e-5, tag="7B_good_10_11"))

    # 7B: second good config
    results.append(run_junction_ft_v2(M7B, 18, 21, num_steps=150, lr=1e-5, tag="7B_good_18_21"))

    # 7B: bad config — can FT rescue it?
    results.append(run_junction_ft_v2(M7B, 4, 9, num_steps=150, lr=1e-5, tag="7B_bad_4_9"))

    # 7B: the proportional config that failed before
    results.append(run_junction_ft_v2(M7B, 15, 18, num_steps=150, lr=1e-5, tag="7B_failed_15_18"))

    # 72B: Ng's config
    results.append(run_junction_ft_v2(M72B, 45, 52, num_steps=75, lr=3e-6, tag="72B_ng_45_52"))

    # 72B: our best config
    results.append(run_junction_ft_v2(M72B, 50, 60, num_steps=75, lr=3e-6, tag="72B_ours_50_60"))

    # Summary
    print(f"\n{'='*80}")
    print("JUNCTION FT V2 — COMPREHENSIVE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Tag':>20} {'Baseline':>9} {'Pre-FT':>9} {'Post-FT':>9} {'FT Gain':>9} {'Verdict':>12}")
    for r in results:
        if r['ft_gain'] > 0.01:
            verdict = "IMPROVED"
        elif r['ft_gain'] > -0.01:
            verdict = "no change"
        else:
            verdict = "hurt"
        print(f"{r['tag']:>20} {r['baseline']:9.4f} {r['pre_ft']:9.4f} "
              f"{r['post_ft']:9.4f} {r['ft_gain']:+9.4f} {verdict:>12}")

    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
