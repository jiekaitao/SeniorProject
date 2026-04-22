"""
lm-eval for Paradigm Shift models.

Loads trained OPLoRA checkpoint, applies K=2 with proper pass-2-only toggle,
optionally adds FFN whisper, then runs lm-eval benchmarks.

Usage:
    python paradigm_lmeval.py --model <path> --checkpoint <path> --tasks bbh,math,mmlu_pro,musr
"""

import os, sys, json, argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from paradigm_shift import (
    Pass2OPLoRA, RecursionGate, LayerIdxWrapper,
    get_inner, inject_oplora, build_k2, restore, setup_ffn_whisper,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


TASK_MAP = {
    'bbh': 'leaderboard_bbh',
    'math': 'leaderboard_math_hard',
    'mmlu_pro': 'leaderboard_mmlu_pro',
    'musr': 'leaderboard_musr',
    'ifeval': 'leaderboard_ifeval',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoint', required=True, help='Path to paradigm shift checkpoint.pt')
    parser.add_argument('--cache_dir', default='/blue/cis4914/jietao/hf_cache')
    parser.add_argument('--tasks', default='bbh,math,mmlu_pro,musr',
                        help='Comma-separated task keys (bbh,math,mmlu_pro,musr,ifeval)')
    parser.add_argument('--limit', type=float, default=0.0,
                        help='Fraction of each task (0=full)')
    parser.add_argument('--ffn_whisper', type=float, default=0.0,
                        help='FFN whisper beta (0=disabled, 0.2=standard whisper)')
    parser.add_argument('--baseline', action='store_true',
                        help='Also run baseline (no dup) for comparison')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    # Load checkpoint config
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    core_start = cfg['core_start']
    core_end = cfg['core_end']
    rank_attn = cfg['rank_attn']
    rank_ffn = cfg['rank_ffn']

    device = torch.device('cuda')

    # Load model
    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, device_map='auto',
        dtype=torch.bfloat16, trust_remote_code=True,
    )
    model_dtype = next(model.parameters()).dtype

    for param in model.parameters():
        param.requires_grad = False

    # Inject OPLoRA and load weights
    print(f'Injecting OPLoRA [core {core_start}-{core_end})...', flush=True)
    lora_modules, layer_loras = inject_oplora(
        model, core_start, core_end,
        rank_attn=rank_attn, rank_ffn=rank_ffn,
        device=device, dtype=model_dtype,
    )

    lora_state = ckpt['lora_state']
    for i, m in enumerate(lora_modules):
        key = f'lora_{i}'
        m.lora_A.data.copy_(lora_state[key]['A'].to(device=device, dtype=model_dtype))
        m.lora_B.data.copy_(lora_state[key]['B'].to(device=device, dtype=model_dtype))
    print(f'  Loaded {len(lora_modules)} LoRA modules from checkpoint', flush=True)

    # Resolve tasks
    task_keys = [t.strip() for t in args.tasks.split(',')]
    task_list = [TASK_MAP.get(k, k) for k in task_keys]
    eval_limit = args.limit if args.limit > 0 else None

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator

    all_results = {}

    # === Baseline (optional) ===
    if args.baseline:
        print(f'\n=== Baseline (K=1, no dup) ===', flush=True)
        for m in lora_modules:
            m.pass_idx = 1
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
        results = evaluator.simple_evaluate(model=lm, tasks=task_list, limit=eval_limit)
        scores = {}
        for task, data in results['results'].items():
            for metric, value in data.items():
                if isinstance(value, (int, float)):
                    scores[f"{task}/{metric}"] = value
        print('\n  BASELINE RESULTS:')
        for k, v in sorted(scores.items()):
            if 'acc' in k or 'exact_match' in k or 'score' in k:
                print(f'    {k}: {v:.4f}')
        all_results['baseline'] = scores

    # === K=2 with trained OPLoRA ===
    print(f'\n=== K=2 with trained OPLoRA ===', flush=True)
    for m in lora_modules:
        m.pass_idx = 1
    orig, orig_N = build_k2(model, core_start, core_end, layer_loras)

    # Optional FFN whisper
    whisper_hooks = []
    if args.ffn_whisper > 0:
        print(f'  FFN whisper beta={args.ffn_whisper}', flush=True)
        whisper_hooks = setup_ffn_whisper(model, core_start, core_end, beta=args.ffn_whisper)

    # Patch generate to disable cache (safety)
    original_generate = model.generate
    def patched_generate(*a, **kw):
        kw['use_cache'] = False
        return original_generate(*a, **kw)
    model.generate = patched_generate

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    results = evaluator.simple_evaluate(model=lm, tasks=task_list, limit=eval_limit)

    scores = {}
    for task, data in results['results'].items():
        for metric, value in data.items():
            if isinstance(value, (int, float)):
                scores[f"{task}/{metric}"] = value

    print('\n  K=2 RESULTS:')
    for k, v in sorted(scores.items()):
        if 'acc' in k or 'exact_match' in k or 'score' in k:
            print(f'    {k}: {v:.4f}')

    tag = f'k2_oplora{"_whisper" + str(args.ffn_whisper) if args.ffn_whisper > 0 else ""}'
    all_results[tag] = scores

    # Cleanup
    for h in whisper_hooks:
        h.remove()
    model.generate = original_generate
    restore(model, orig, orig_N)

    # === Delta summary ===
    if args.baseline:
        print('\n  DELTAS (K=2 vs Baseline):')
        for k in sorted(all_results[tag].keys()):
            if k in all_results['baseline']:
                delta = all_results[tag][k] - all_results['baseline'][k]
                if 'acc' in k or 'exact_match' in k or 'score' in k:
                    print(f'    {k}: {delta:+.4f}')

    # Save
    output_path = args.output or f'sirt/recursion_ft/{cfg["name"]}_paradigm/lmeval_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'model': args.model,
            'checkpoint': args.checkpoint,
            'core': [core_start, core_end],
            'tasks': task_list,
            'limit': args.limit,
            'ffn_whisper': args.ffn_whisper,
            'results': all_results,
        }, f, indent=2)
    print(f'\nSaved to {output_path}')


if __name__ == '__main__':
    main()
