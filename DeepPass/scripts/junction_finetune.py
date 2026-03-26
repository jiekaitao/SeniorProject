"""
DeepPass Junction Fine-Tuning

Ng's key hypothesis: "A little bit of fine-tuning on those two junction layers
is all we really need." The junction is where duplicated block output feeds
back into the block input — a distributional shift the model never saw in training.

This script:
1. Freezes all layers EXCEPT the junction layers
2. Fine-tunes only those 2 layers on a small dataset
3. Tests whether this improves the duplicated model

The junction layers for config (i=45, j=52) on an 87-layer model are:
- Layer 51 (end of first pass, outputs to...)
- Layer 45' (= layer 52 in the 87-layer model, start of second pass)
These are the transition points where out-of-distribution hidden states occur.
"""

import sys, os, json, torch, copy
import torch.nn as nn
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import load_original_model, apply_layer_duplication, generate_no_cache
from math_probe import run_math_probe
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = Path(__file__).parent.parent / "results"


def get_junction_layer_indices(i, j, N):
    """
    Get the indices of junction layers in the duplicated model.

    For config (i, j) in the duplicated model:
    - Original path: [0, 1, ..., j-1, i, i+1, ..., j-1, j, ..., N-1]
    - The junction is at position j-1 (end of first block) → position j (start of dup block = layer i)
    - In the new numbering (after duplication):
      - End of first block: layer index j-1
      - Start of duplicated block: layer index j (which is a copy of original layer i)
    """
    # In the 87-layer model:
    # Layers 0-51 are original layers 0-51
    # Layers 52-58 are copies of original layers 45-51
    # Layers 59-86 are original layers 52-79
    #
    # Junction: layer 51 (end first pass) → layer 52 (start dup, copy of 45)
    end_first_pass = j - 1
    start_dup_pass = j  # This is the copy of layer i
    end_dup_pass = j + (j - i) - 1  # This is the copy of layer j-1
    start_remaining = j + (j - i)  # This is original layer j

    return {
        "end_first_pass": end_first_pass,
        "start_dup_pass": start_dup_pass,
        "end_dup_pass": end_dup_pass,
        "start_remaining": start_remaining,
    }


def finetune_junction(
    model_path: str,
    i: int = 45,
    j: int = 52,
    num_steps: int = 100,
    lr: float = 1e-5,
    output_dir: str = None,
):
    """Fine-tune only the junction layers of a duplicated model."""
    if output_dir is None:
        output_dir = RESULTS_DIR / f"junction_ft_{Path(model_path).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and applying duplication...")
    model, tokenizer = load_original_model(model_path)

    # Deep copy layers for duplication so we can fine-tune independently
    inner = model.model
    layers = list(inner.layers)
    N = len(layers)

    new_layers = layers[:j]
    for idx in range(i, j):
        new_layers.append(copy.deepcopy(layers[idx]))
    new_layers.extend(layers[j:])

    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    new_N = len(new_layers)

    junction = get_junction_layer_indices(i, j, N)
    print(f"Model: {N} → {new_N} layers")
    print(f"Junction layers: {junction}")

    # Pre-finetune benchmark
    print("\n--- Pre-finetune math probe ---")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    pre_result = run_math_probe(gen_fn, verbose=True)
    print(f"Pre-finetune score: {pre_result['score']:.4f}")

    # Freeze everything except junction layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze junction layers
    junction_indices = [
        junction["end_first_pass"],
        junction["start_dup_pass"],
        junction["end_dup_pass"],
        junction["start_remaining"],
    ]

    trainable_params = 0
    for idx in junction_indices:
        if idx < new_N:
            for param in inner.layers[idx].parameters():
                param.requires_grad = True
                trainable_params += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")

    # Simple training data — we use the model's own generations as a self-distillation signal
    # The idea: run the ORIGINAL model to get "good" outputs, then train the
    # duplicated model's junction layers to better reproduce those outputs
    training_prompts = [
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
        "A linked list is a data structure where",
        "The speed of light in a vacuum is approximately",
        "Mitochondria are known as the powerhouses of",
        "The capital of Japan is",
        "Neural networks consist of layers of",
        "The integral of 1/x is",
    ]

    # Get target logits from the model (self-distillation)
    print("\nGenerating training targets...")
    model.eval()
    targets = []
    for prompt in training_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True,
                          truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
        targets.append((inputs, outputs.logits.detach()))

    # Fine-tune
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )

    print(f"\nFine-tuning junction layers for {num_steps} steps...")
    model.train()
    losses = []

    for step in range(num_steps):
        total_loss = 0
        for inputs, target_logits in targets:
            outputs = model(**inputs, use_cache=False)
            # KL divergence between current and target logits
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(outputs.logits[:, :-1, :], dim=-1),
                nn.functional.softmax(target_logits[:, :-1, :], dim=-1),
                reduction='batchmean'
            )
            loss.backward()
            total_loss += loss.item()

        avg_loss = total_loss / len(targets)
        losses.append(avg_loss)

        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{num_steps} loss={avg_loss:.6f}")

    # Post-finetune benchmark
    print("\n--- Post-finetune math probe ---")
    model.eval()
    post_result = run_math_probe(gen_fn, verbose=True)
    print(f"Post-finetune score: {post_result['score']:.4f}")

    # Compare
    delta = post_result['score'] - pre_result['score']
    print(f"\n{'='*60}")
    print(f"JUNCTION FINE-TUNING RESULTS")
    print(f"  Pre-finetune:  {pre_result['score']:.4f}")
    print(f"  Post-finetune: {post_result['score']:.4f}")
    print(f"  Delta:         {delta:+.4f}")
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.4f}%)")
    print(f"{'='*60}")

    results = {
        "pre_score": pre_result['score'],
        "post_score": post_result['score'],
        "delta": delta,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "num_steps": num_steps,
        "lr": lr,
        "losses": losses,
        "junction_indices": junction_indices,
    }
    with open(output_dir / "junction_ft_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct")
    parser.add_argument("--i", type=int, default=None)
    parser.add_argument("--j", type=int, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    # Default to proportional config if not specified
    if args.i is None or args.j is None:
        # Will be set based on model layer count inside finetune_junction
        model, tok = load_original_model(args.model)
        inner = model.model if hasattr(model, 'model') else model.transformer
        layers = inner.layers if hasattr(inner, 'layers') else inner.h
        N = len(layers)
        args.i = int(N * 0.5625)
        args.j = int(N * 0.65)
        if args.j <= args.i:
            args.j = args.i + 2
        del model
        torch.cuda.empty_cache()

    finetune_junction(args.model, i=args.i, j=args.j, num_steps=args.steps, lr=args.lr)
