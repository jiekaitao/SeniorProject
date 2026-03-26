"""
DeepPass Junction Fine-Tuning on 72B

Ng's hypothesis: "A little bit of fine-tuning on those two layers is all we
really need." He never tested this. We will.

Strategy:
1. Load the duplicated 87-layer model
2. Freeze everything except the 2 junction layers (where the loop-back happens)
3. Fine-tune on a small self-distillation dataset
4. Measure improvement on math probe

The junction in config (45,52) on the 87-layer model:
- Layer 51 (last layer of first pass through block)
- Layer 52 (first layer of second pass = copy of original layer 45)
These see out-of-distribution hidden states at the seam.
"""

import sys, os, json, copy, torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe


def main():
    MODEL_PATH = "/blue/cis4914/jietao/DeepPass/models/full/calme-2.1-qwen2-72b"
    RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/junction_ft_72b")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    I, J = 45, 52  # Ng's config

    print("Loading 72B model...")
    model, tokenizer = load_original_model(MODEL_PATH)
    inner = model.model
    layers = list(inner.layers)
    N = len(layers)

    # Build duplicated model with deep copies at junction
    print(f"Building duplicated model with junction deep copies...")
    new_layers = layers[:J]
    for idx in range(I, J):
        new_layers.append(copy.deepcopy(layers[idx]))
    new_layers.extend(layers[J:])

    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    new_N = len(new_layers)
    print(f"Model: {N} → {new_N} layers")

    # Junction layer indices in the 87-layer model
    junction_layers = [J - 1, J]  # 51 and 52
    print(f"Junction layers: {junction_layers}")

    # Pre-FT benchmark
    print("\n--- Pre-finetune math probe ---")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    pre_result = run_math_probe(gen_fn, verbose=True)
    print(f"Pre-FT score: {pre_result['score']:.4f}")

    # Freeze everything except junction layers
    for param in model.parameters():
        param.requires_grad = False

    trainable = 0
    for idx in junction_layers:
        for param in inner.layers[idx].parameters():
            param.requires_grad = True
            trainable += param.numel()

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    # Training prompts
    prompts = [
        "The theory of general relativity states that",
        "In Python, a decorator is a function that",
        "The process of photosynthesis involves",
        "To solve a quadratic equation, you can use",
        "The French Revolution began in the year",
        "Machine learning models are trained by",
        "The derivative of sin(x) is",
        "In economics, supply and demand determine",
        "DNA replication begins when the enzyme",
        "The Pythagorean theorem states that",
        "Neural networks consist of layers of",
        "The integral of 1/x is",
        "A binary search tree is a data structure",
        "The speed of light in a vacuum is",
        "Mitochondria are known as the powerhouses",
        "The capital of Japan is",
        "What is 78313 multiplied by 88537?",
        "The cube root of 74088 is approximately",
        "What is 9999 multiplied by 9999?",
        "The square root of 152399025 is",
    ]

    # Get target logits (self-distillation)
    print("\nGenerating training targets...")
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
        lr=5e-6, weight_decay=0.01
    )

    print(f"\nFine-tuning junction layers for 50 steps...")
    model.train()
    losses = []

    for step in range(50):
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

        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/50 loss={avg:.6f}")

    # Post-FT benchmark
    print("\n--- Post-finetune math probe ---")
    model.eval()
    post_result = run_math_probe(gen_fn, verbose=True)
    print(f"Post-FT score: {post_result['score']:.4f}")

    delta = post_result['score'] - pre_result['score']
    print(f"\n{'='*60}")
    print(f"JUNCTION FINE-TUNING RESULTS (72B)")
    print(f"  Pre-FT:   {pre_result['score']:.4f}")
    print(f"  Post-FT:  {post_result['score']:.4f}")
    print(f"  Delta:    {delta:+.4f}")
    print(f"  Params:   {trainable:,} ({100*trainable/total:.4f}%)")
    print(f"{'='*60}")

    results = {
        "pre_score": pre_result['score'],
        "post_score": post_result['score'],
        "delta": delta,
        "trainable_params": trainable,
        "total_params": total,
        "losses": losses,
    }
    with open(RESULTS_DIR / "junction_ft_72b_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
