"""
Run lm-eval on a model with runtime layer duplication.
No model saving needed — applies duplication in-memory then evaluates.
"""
import argparse
import sys
import os
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from layer_duplicator import load_original_model


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--blocks", required=True, help="Blocks as 'i1,j1;i2,j2;...'")
    parser.add_argument("--tasks", default="leaderboard_bbh,leaderboard_mmlu_pro")
    parser.add_argument("--limit", type=float, default=0.15)
    parser.add_argument("--output", default="results/lm_eval_dup.json")
    args = parser.parse_args()

    # Parse blocks
    blocks = []
    for part in args.blocks.split(";"):
        i, j = part.strip().split(",")
        blocks.append((int(i), int(j)))
    print(f"Blocks to duplicate: {blocks}")

    # Load model
    model, tokenizer = load_original_model(args.model)
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)

    # Apply duplication
    order = build_layer_order(blocks, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)

    # Fix layer_types if present
    if hasattr(model.config, 'layer_types') and model.config.layer_types:
        orig_types = model.config.layer_types
        new_types = [orig_types[idx % len(orig_types)] for idx in order]
        model.config.layer_types = new_types

    print(f"Applied duplication: {N} -> {len(order)} layers")

    # Wrap model for lm-eval using HFLM
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator

    # Create HFLM wrapper with our already-loaded model
    # 72B uses ~160GB; only ~10GB headroom on B200 (179GB)
    # batch_size=1 is safe; "auto" OOMs at ~165 loglikelihood requests
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,
    )

    # Disable cache for duplicated model
    original_generate = model.generate

    def patched_generate(*a, **kw):
        kw['use_cache'] = False
        return original_generate(*a, **kw)

    model.generate = patched_generate

    # Run evaluation
    task_list = args.tasks.split(",")
    print(f"Tasks: {task_list}")
    print(f"Limit: {args.limit}")

    eval_limit = args.limit if args.limit > 0 else None
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        limit=eval_limit,
    )

    # Extract scores
    scores = {}
    for task, data in results['results'].items():
        for metric, value in data.items():
            if isinstance(value, (int, float)):
                scores[f"{task}/{metric}"] = value

    # Print summary
    print("\n=== RESULTS ===")
    for k, v in sorted(scores.items()):
        print(f"  {k}: {v:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'model': args.model,
            'blocks': [list(b) for b in blocks],
            'tasks': task_list,
            'limit': args.limit,
            'scores': scores,
            'results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))}
                        for k, v in results['results'].items()},
        }, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
