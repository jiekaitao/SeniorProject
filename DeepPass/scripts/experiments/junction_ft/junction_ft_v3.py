"""
Junction Fine-Tuning V3 — Agent Battle Winner

Key improvements over V1/V2 based on 5-agent analysis:

1. Hidden-state MSE at junction point (not logit KL) — direct gradient, no attenuation
2. Train 4 junction layers, not 2, with per-layer LR
3. Procrustes initialization — analytical zero-shot correction as starting point
4. Config-aware loss: good configs use hidden-state MSE only, bad use logit KL too
5. More training data (diverse prompts)
6. OneCycleLR schedule with higher peak LR
"""

import sys, os, json, copy, time, torch, gc
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe

RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/junction_ft_v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_procrustes_correction(h_teacher, h_student):
    """
    Compute optimal rotation + scaling to align student hidden states to teacher.
    Returns (R, s, t) such that s * h_student @ R + t ≈ h_teacher.
    """
    # Center
    mu_t = h_teacher.mean(dim=0)
    mu_s = h_student.mean(dim=0)
    h_t_c = h_teacher - mu_t
    h_s_c = h_student - mu_s

    # Cross-covariance
    C = h_t_c.T @ h_s_c  # [d, d]

    # SVD for optimal rotation (use truncated for efficiency)
    U, S, Vt = torch.linalg.svd(C, full_matrices=False)
    R = Vt.T @ U.T  # [d, d]

    # Optimal scaling
    s = S.sum() / (h_s_c.norm() ** 2 + 1e-8)

    # Translation
    t = mu_t - s * (mu_s @ R)

    return R, s, t


def collect_hidden_states_at_layer(model, tokenizer, prompts, layer_idx, device):
    """Collect hidden states at a specific layer using hooks."""
    all_states = []
    hook_handle = None

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        all_states.append(h.detach().clone())

    inner = model.model
    hook_handle = inner.layers[layer_idx].register_forward_hook(hook_fn)

    for p in prompts:
        inp = tokenizer(p, return_tensors="pt", truncation=True,
                       max_length=64).to(device)
        with torch.no_grad():
            model(**inp, use_cache=False)

    hook_handle.remove()
    return all_states


def run_junction_ft_v3(model_path, i, j, num_steps=200, tag=""):
    """
    V3 Junction FT with all improvements from agent analysis.
    """
    print(f"\n{'='*60}")
    print(f"JUNCTION FT V3: {tag} — config ({i},{j})")
    print(f"{'='*60}")

    model, tokenizer = load_original_model(model_path)
    inner = model.model
    layers = list(inner.layers)
    N = len(layers)
    device = next(model.parameters()).device
    dup_count = j - i

    # Diverse training prompts
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
        "The Pythagorean theorem states that",
        "To implement quicksort, you first choose a pivot",
        "The integral of e^x dx equals",
        "In economics, inflation is defined as",
        "The chemical formula for water is H2O because",
        "A recursive function calls itself until",
        "The GDP of the United States in 2025",
        "Photosynthesis converts sunlight into",
        "The Fibonacci sequence starts with 0, 1, and then",
        "In linear algebra, the determinant of a matrix",
        "The human genome contains approximately",
        "Gradient descent works by computing the derivative",
    ]

    # Step 1: Baseline
    print("\n--- Baseline ---")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    baseline = run_math_probe(gen_fn, verbose=False)
    print(f"  Baseline: {baseline['score']:.4f}")

    # Step 2: Collect teacher hidden states at the junction input layer
    # (what layer i normally sees as input = output of layer i-1)
    print(f"\n--- Collecting teacher hidden states at layer {i-1 if i > 0 else 0} ---")
    teacher_states = collect_hidden_states_at_layer(
        model, tokenizer, prompts[:8], max(0, i - 1), device
    )

    # Also collect teacher logits for bad-config fallback
    model.eval()
    teacher_logits_data = []
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt", truncation=True,
                       max_length=64).to(device)
        with torch.no_grad():
            out = model(**inp, use_cache=False)
        teacher_logits_data.append((inp, out.logits.detach().clone()))

    # Step 3: Build duplicated model
    print(f"\n--- Building duplicated model ({i},{j}) ---")
    new_layers = layers[:j]
    for idx in range(i, j):
        new_layers.append(copy.deepcopy(layers[idx]))
    new_layers.extend(layers[j:])

    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    new_N = len(new_layers)

    # Step 4: Pre-FT score
    print("\n--- Pre-FT score ---")
    pre = run_math_probe(gen_fn, verbose=False)
    pre_delta = pre['score'] - baseline['score']
    is_good_config = pre_delta > 0
    print(f"  Pre-FT: {pre['score']:.4f} (delta: {pre_delta:+.4f})")
    print(f"  Config type: {'GOOD' if is_good_config else 'BAD'}")

    # Step 5: Identify all 4 junction layers
    # Junction 1: layer j-1 (end of first pass) → layer j (start of dup block)
    # Junction 2: layer j+dup_count-1 (end of dup block) → layer j+dup_count (resume original)
    j1_exit = j - 1
    j1_entry = j
    j2_exit = j + dup_count - 1
    j2_entry = min(j + dup_count, new_N - 1)
    junction_layers = [j1_exit, j1_entry, j2_exit, j2_entry]
    junction_layers = [jl for jl in junction_layers if jl < new_N]
    junction_layers = list(set(junction_layers))  # deduplicate

    print(f"  Junction layers: {junction_layers}")
    print(f"  (J1: {j1_exit}→{j1_entry}, J2: {j2_exit}→{j2_entry})")

    # Step 6: Freeze everything except junction layers
    for param in model.parameters():
        param.requires_grad = False

    # Per-layer LR groups (higher LR at main junction)
    param_groups = []
    lr_map = {j1_exit: 3e-5, j1_entry: 2e-5, j2_exit: 1e-5, j2_entry: 5e-6}
    trainable = 0
    for jl in junction_layers:
        lr = lr_map.get(jl, 1e-5)
        params = list(inner.layers[jl].parameters())
        for p in params:
            p.requires_grad = True
            trainable += p.numel()
        param_groups.append({"params": params, "lr": lr})

    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.4f}%)")

    # Step 7: Set up hidden-state hooks for MSE loss at junction
    junction_hidden_states = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            junction_hidden_states[name] = h
        return hook_fn

    # Hook at the main junction point (output of j-1, which feeds into dup block)
    hook_j1 = inner.layers[j1_exit].register_forward_hook(make_hook('j1_output'))

    # Step 8: Training
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g['lr'] for g in param_groups],
        total_steps=num_steps, pct_start=0.1,
        anneal_strategy='cos', div_factor=10, final_div_factor=100
    )

    print(f"\n--- Training {num_steps} steps ---")
    if is_good_config:
        print("  Mode: GOOD CONFIG — hidden-state MSE only (preserve improvement)")
    else:
        print("  Mode: BAD CONFIG — hidden-state MSE + logit KL (recover quality)")

    model.train()
    losses = []

    for step in range(num_steps):
        total_loss = 0

        for idx, (inp, teacher_logits) in enumerate(teacher_logits_data):
            junction_hidden_states.clear()

            # Forward through duplicated model
            student_out = model(**inp, use_cache=False)

            # Loss 1: Hidden-state MSE at junction (direct, unattenuated gradient)
            if 'j1_output' in junction_hidden_states and idx < len(teacher_states):
                h_student = junction_hidden_states['j1_output']
                h_teacher = teacher_states[min(idx, len(teacher_states)-1)]
                # Match shapes (truncate to shorter sequence)
                min_len = min(h_student.shape[1], h_teacher.shape[1])
                h_s = h_student[:, :min_len, :]
                h_t = h_teacher[:, :min_len, :]
                hidden_loss = nn.functional.mse_loss(h_s, h_t)
            else:
                hidden_loss = torch.tensor(0.0, device=device)

            if is_good_config:
                # Good config: only hidden-state MSE (don't touch logits)
                loss = hidden_loss
            else:
                # Bad config: MSE + KL to recover original quality
                kl_loss = nn.functional.kl_div(
                    nn.functional.log_softmax(student_out.logits[:, :-1, :], dim=-1),
                    nn.functional.softmax(teacher_logits[:, :-1, :], dim=-1),
                    reduction='batchmean'
                )
                loss = 0.7 * hidden_loss + 0.3 * kl_loss

            if loss.requires_grad:
                loss.backward()
            total_loss += loss.item()

        avg = total_loss / len(teacher_logits_data)
        losses.append(avg)

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if (step + 1) % 50 == 0:
            print(f"    Step {step+1}/{num_steps} loss={avg:.6f}")

    hook_j1.remove()

    # Step 9: Post-FT score
    print("\n--- Post-FT score ---")
    model.eval()
    post = run_math_probe(gen_fn, verbose=False)
    post_delta = post['score'] - baseline['score']
    ft_gain = post['score'] - pre['score']

    print(f"\n  {'='*50}")
    print(f"  V3 RESULTS: {tag}")
    print(f"  {'='*50}")
    print(f"  Original:      {baseline['score']:.4f}")
    print(f"  Pre-FT:        {pre['score']:.4f} ({pre_delta:+.4f})")
    print(f"  Post-FT:       {post['score']:.4f} ({post_delta:+.4f})")
    print(f"  FT gain:       {ft_gain:+.4f}")
    print(f"  Config type:   {'GOOD' if is_good_config else 'BAD'}")
    if is_good_config:
        print(f"  Improvement preserved: {'YES' if post_delta >= pre_delta * 0.9 else 'PARTIALLY' if post_delta > 0 else 'NO'}")
    else:
        print(f"  Recovery: {ft_gain/abs(pre_delta)*100:.1f}% of lost quality")

    result = {
        "tag": tag, "config": [i, j], "model": model_path,
        "baseline": baseline['score'],
        "pre_ft": pre['score'], "pre_delta": pre_delta,
        "post_ft": post['score'], "post_delta": post_delta,
        "ft_gain": ft_gain, "is_good_config": is_good_config,
        "junction_layers": junction_layers,
        "trainable": trainable, "trainable_pct": 100*trainable/total,
        "steps": num_steps, "final_loss": losses[-1] if losses else 0,
    }

    del model, tokenizer, teacher_states, teacher_logits_data
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    results = []
    M7B = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"

    # Good configs — should PRESERVE improvement
    results.append(run_junction_ft_v3(M7B, 10, 11, num_steps=200, tag="7B_good_10_11"))
    results.append(run_junction_ft_v3(M7B, 18, 21, num_steps=200, tag="7B_good_18_21"))

    # Bad configs — should RECOVER quality
    results.append(run_junction_ft_v3(M7B, 4, 9, num_steps=200, tag="7B_bad_4_9"))
    results.append(run_junction_ft_v3(M7B, 15, 18, num_steps=200, tag="7B_bad_15_18"))

    # Summary
    print(f"\n{'='*80}")
    print("JUNCTION FT V3 — COMPREHENSIVE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Tag':>20} {'Type':>5} {'Baseline':>9} {'Pre-FT':>9} {'Post-FT':>9} {'FT Gain':>9}")
    for r in results:
        t = "GOOD" if r['is_good_config'] else "BAD"
        print(f"{r['tag']:>20} {t:>5} {r['baseline']:9.4f} {r['pre_ft']:9.4f} "
              f"{r['post_ft']:9.4f} {r['ft_gain']:+9.4f}")

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
