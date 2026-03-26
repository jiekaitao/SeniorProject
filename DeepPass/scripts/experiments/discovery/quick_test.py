"""
Quick sanity test — verify layer duplication works on a small model.
Run on Qwen2-7B-Instruct before committing to the full 72B sweep.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import load_original_model, apply_layer_duplication, generate_no_cache
from math_probe import run_math_probe


def make_generate_fn(model, tokenizer, use_cache=True):
    """Create a generate function. use_cache=False for duplicated models."""
    if use_cache:
        def generate_fn(prompt):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=64, do_sample=False,
                )
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True)
        return generate_fn
    else:
        def generate_fn(prompt):
            return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
        return generate_fn


def main():
    model_path = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"

    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Model not found at {model_path}")
        sys.exit(1)

    # 1. Test baseline (with cache for speed)
    print("=" * 60)
    print("TEST 1: Baseline (no duplication)")
    print("=" * 60)
    model, tokenizer = load_original_model(model_path)

    inner = model.model if hasattr(model, 'model') else model.transformer
    layers = inner.layers if hasattr(inner, 'layers') else inner.h
    num_layers = len(layers)
    original_layers = list(layers)
    print(f"Model has {num_layers} layers")

    gen_fn = make_generate_fn(model, tokenizer, use_cache=True)
    baseline = run_math_probe(gen_fn, verbose=True)
    print(f"\nBaseline score: {baseline['score']:.4f}")

    # Sanity check
    test_prompt = "System: Answer with only the number.\n\nUser: What is 2 + 2? Answer with only the number:\n\nAssistant:"
    response = gen_fn(test_prompt)
    print(f"Sanity check: 2+2 = {response.strip()}")

    # 2. Test with layer duplication (no cache)
    # For 28-layer Qwen2-7B, proportional to Ng's 45/52 on 80-layer:
    # 45/80 ≈ 0.5625 → 16, 52/80 = 0.65 → 18
    i = int(num_layers * 0.5625)
    j = int(num_layers * 0.65)
    if j <= i:
        j = i + 2

    print(f"\n{'=' * 60}")
    print(f"TEST 2: Duplicated layers ({i}, {j}) — {j - i} layers duplicated")
    print("=" * 60)

    apply_layer_duplication(model, i, j)

    # Use no-cache generation for duplicated model
    gen_fn_dup = make_generate_fn(model, tokenizer, use_cache=False)
    dup_result = run_math_probe(gen_fn_dup, verbose=True)
    print(f"\nDuplicated score: {dup_result['score']:.4f}")

    # 3. Compare
    delta = dup_result['score'] - baseline['score']
    pct = (delta / baseline['score'] * 100) if baseline['score'] > 0 else float('inf')
    print(f"\n{'=' * 60}")
    print(f"COMPARISON")
    print(f"  Baseline:    {baseline['score']:.4f}")
    print(f"  Duplicated:  {dup_result['score']:.4f}")
    print(f"  Delta:       {delta:+.4f} ({pct:+.2f}%)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
