"""Unified PEFT-variants runner: vanilla LoRA, LoRA+, DoRA, rsLoRA.

Tests whether vanilla LoRA's collapse on SpatialGrid is fundamental to
static low-rank PEFT or just a LoRA-specific parameterization problem.

All methods use ~140M trainable params (rank 64 on all 7 projections × 32 layers).

Adds diagnostics every 500 steps:
  - CCS (Counterfactual Count Sensitivity): variance of choice logits
    when the target animal is swapped but grid kept fixed. If CCS→0,
    the model is collapsing to a non-counting shortcut.
  - Prediction entropy on probe set.
  - Modal rate: fraction predicting the most common letter.
"""
import os, sys, torch, json, random, math, time, argparse, re
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))

device = torch.device('cuda')
CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
SPATIALGRID_ANIMALS = ['cat', 'dog', 'elephant', 'giraffe', 'rabbit']


def make_peft_config(method, rank, use_bias='none', dropout=0.05):
    """Build LoraConfig for the requested method."""
    base = dict(
        task_type=TaskType.CAUSAL_LM, r=rank,
        lora_alpha=rank * 2, lora_dropout=dropout, bias=use_bias,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    if method == 'lora':
        return LoraConfig(**base)
    elif method == 'lora_plus':
        return LoraConfig(**base)  # same config; lr split happens at optimizer
    elif method == 'dora':
        return LoraConfig(**base, use_dora=True)
    elif method == 'rslora':
        return LoraConfig(**base, use_rslora=True)
    else:
        raise ValueError(f'Unknown PEFT method: {method}')


def build_optimizer(model, method, lr, weight_decay, lora_plus_ratio):
    """Build AdamW. For LoRA+, A and B get different learning rates."""
    if method != 'lora_plus':
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    # LoRA+: lr_B = ratio * lr_A
    params_A, params_B, params_other = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'lora_A' in n:
            params_A.append(p)
        elif 'lora_B' in n:
            params_B.append(p)
        else:
            # magnitude (DoRA) or any other trainable — put with B
            params_other.append(p)
    groups = [
        {'params': params_A, 'lr': lr, 'name': 'lora_A'},
        {'params': params_B, 'lr': lr * lora_plus_ratio, 'name': 'lora_B'},
    ]
    if params_other:
        groups.append({'params': params_other, 'lr': lr, 'name': 'other'})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def compute_ccs_and_entropy(model, tokenizer, probe_samples, choice_ids_t):
    """Diagnostic: counterfactual count sensitivity + entropy + modal rate.
    probe_samples: list of {'text': raw text with target animal, 'target_animal': str}.
    """
    if not probe_samples:
        return {'ccs': None, 'entropy': None, 'modal_rate': None}
    model.eval()
    all_swap_probs = []      # shape per probe: (n_animals, n_choices)
    all_native_probs = []    # shape per probe: (n_choices,)
    all_preds = []
    with torch.no_grad():
        for sample in probe_samples:
            text = sample['text']
            target = sample['target_animal']
            # Compute probs for each animal swap
            swap_probs = []
            for animal in SPATIALGRID_ANIMALS:
                swapped = text.replace(f'contain {target}', f'contain {animal}')
                swapped_full = swapped[:1500] + '\nAnswer:'
                enc = tokenizer(swapped_full, return_tensors='pt',
                                truncation=True, max_length=1900).to(device)
                out = model(enc['input_ids'])
                logits = out.logits[:, -1, choice_ids_t]
                probs = logits.softmax(dim=-1).squeeze(0).float().cpu()
                swap_probs.append(probs)
                if animal == target:
                    all_native_probs.append(probs)
                    all_preds.append(int(probs.argmax().item()))
            all_swap_probs.append(torch.stack(swap_probs, dim=0))  # (n_animals, n_choices)
    # CCS = mean variance across swaps
    import torch as _t
    swap_t = _t.stack(all_swap_probs, dim=0)  # (n_probes, n_animals, n_choices)
    ccs = swap_t.var(dim=1).mean().item()
    native_t = _t.stack(all_native_probs, dim=0)  # (n_probes, n_choices)
    entropy = -(native_t * (native_t + 1e-10).log()).sum(dim=-1).mean().item()
    # Modal rate
    from collections import Counter
    counts = Counter(all_preds)
    modal = max(counts.values()) / len(all_preds) if all_preds else 0
    model.train()
    return {'ccs': ccs, 'entropy': entropy, 'modal_rate': modal}


def extract_target_animal(text):
    """Parse 'How many blocks contain X?' to find target animal."""
    m = re.search(r'contain\s+(\w+)', text)
    return m.group(1) if m else None


def build_probe_set(data, eval_idx, n=50):
    """Pick probe samples for CCS: SpatialGrid examples with identifiable target."""
    probes = []
    for idx in eval_idx[:n * 2]:  # oversample
        sample = data[idx]
        text = sample['text']
        target = extract_target_animal(text)
        if target and target in SPATIALGRID_ANIMALS:
            probes.append({'text': text, 'target_animal': target, 'oracle': sample['oracle_option']})
        if len(probes) >= n:
            break
    return probes


def train_and_eval_peft(base_model, tokenizer, data, train_idx, eval_idx,
                        method, rank, total_steps, seed, grad_accum,
                        tag, results_dir, task_name, lr, lora_plus_ratio):
    random.seed(seed)
    torch.manual_seed(seed)
    choice_letters = ['A', 'B', 'C', 'D']
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in choice_letters]
    choice_ids_t = torch.tensor(choice_ids, device=device)

    lora_config = make_peft_config(method, rank)
    model = get_peft_model(base_model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  method={method} r={rank}: {trainable_params/1e6:.2f}M trainable', flush=True)

    optimizer = build_optimizer(model, method, lr, 0.05, lora_plus_ratio)
    warmup = 200
    def lr_sched(s):
        if s < warmup: return s / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    # Baseline
    model.eval()
    print(f'  Computing baseline...', flush=True)
    baseline_correct = 0
    with torch.no_grad():
        for idx in eval_idx:
            sample = data[idx]
            text = sample['text'][:1500] + "\nAnswer:"
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
            out = model(enc['input_ids'])
            logits = out.logits[:, -1, choice_ids_t]
            pred = logits.argmax(dim=-1).item()
            oracle = sample['oracle_option'].strip().upper()
            al = CHOICE_MAP.get(oracle[0], 0)
            if pred == al:
                baseline_correct += 1
    baseline_acc = baseline_correct / len(eval_idx)
    print(f'  Baseline: {baseline_acc:.4f}', flush=True)

    # Build probe set (only makes sense for SpatialGrid)
    probe_set = build_probe_set(data, eval_idx, n=50) if 'spatialgrid' in task_name else []
    print(f'  Probe set size: {len(probe_set)}', flush=True)

    # Train
    model.train()
    t0 = time.time()
    losses = []
    diagnostics = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(total_steps):
        sample = data[train_idx[step % len(train_idx)]]
        text = sample['text'][:1500] + "\nAnswer:"
        oracle = sample['oracle_option'].strip().upper()
        al = CHOICE_MAP.get(oracle[0], 0)
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
        out = model(enc['input_ids'])
        logits = out.logits[:, -1, choice_ids_t]
        label_t = torch.tensor([al], device=device, dtype=torch.long)
        loss = F.cross_entropy(logits.float(), label_t) / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item() * grad_accum)
        if (step + 1) % 500 == 0:
            avg = sum(losses[-500:]) / 500
            diag_msg = ''
            if probe_set:
                diag = compute_ccs_and_entropy(model, tokenizer, probe_set, choice_ids_t)
                diag['step'] = step + 1
                diag['ce'] = avg
                diagnostics.append(diag)
                diag_msg = f" | ccs={diag['ccs']:.4f} ent={diag['entropy']:.3f} modal={diag['modal_rate']:.2f}"
            print(f'  step {step+1} | ce={avg:.4f} | {time.time()-t0:.0f}s{diag_msg}', flush=True)

    # Final eval
    model.eval()
    correct = 0
    with torch.no_grad():
        for idx in eval_idx:
            sample = data[idx]
            text = sample['text'][:1500] + "\nAnswer:"
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
            out = model(enc['input_ids'])
            logits = out.logits[:, -1, choice_ids_t]
            oracle = sample['oracle_option'].strip().upper()
            al = CHOICE_MAP.get(oracle[0], 0)
            if logits.argmax(dim=-1).item() == al: correct += 1
    final_acc = correct / len(eval_idx)
    delta = final_acc - baseline_acc
    print(f'  {method}: {final_acc:.4f} (delta={delta:+.4f})', flush=True)

    os.makedirs(results_dir, exist_ok=True)
    result_data = {
        'tag': tag, 'benchmark': task_name, 'method': method, 'rank': rank,
        'trainable_params_M': trainable_params / 1e6,
        'total_steps': total_steps, 'seed': seed, 'grad_accum': grad_accum,
        'lora_plus_ratio': lora_plus_ratio if method == 'lora_plus' else None,
        'results': {
            'baseline': {'accuracy': baseline_acc, 'correct': baseline_correct, 'total': len(eval_idx)},
            'final': {'accuracy': final_acc, 'correct': correct, 'total': len(eval_idx), 'delta': delta},
        },
        'diagnostics': diagnostics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(results_dir, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved {tag}.json ({time.time()-t0:.0f}s)', flush=True)
    model = model.unload()
    torch.cuda.empty_cache()
    return result_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        choices=['lora', 'lora_plus', 'dora', 'rslora'])
    parser.add_argument('--model', type=str, default='models/full/Llama-3.1-8B-Instruct')
    parser.add_argument('--tasks', type=str, default='spatialgrid')
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--total_steps', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lora_plus_ratio', type=float, default=16.0)
    parser.add_argument('--results_dir', type=str, default='results/data/peft_variants')
    args = parser.parse_args()

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    print(f'Model loaded.', flush=True)

    from datasets import load_dataset
    from mega_runner_benchmarks import BENCHMARK_LOADERS
    spatial_ds = None
    model_short = 'inst' if 'instruct' in args.model.lower() else 'base'

    overall_t0 = time.time()
    for task in args.tasks.split(','):
        task = task.strip()
        print(f'\n=== {args.method.upper()} on {task} ===', flush=True)

        # Route to the right loader
        if task in ('spatialgrid', 'mazenav', 'spatialmap', 'spatialreal'):
            if spatial_ds is None:
                spatial_ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
            data = [s for s in spatial_ds if s['id'].startswith(task)]
            random.seed(0)
            indices = list(range(len(data)))
            random.shuffle(indices)
            split = min(1000, len(indices) * 2 // 3)
            train_idx, eval_idx = indices[:split], indices[split:]
            # data[i]['oracle_option'] stays in existing format
        elif task in BENCHMARK_LOADERS:
            fmt, n_choices = BENCHMARK_LOADERS[task]()
            # Convert to SpatialEval-compatible format for the trainer
            # trainer uses sample['text'] and sample['oracle_option'] (first letter -> index)
            letters = ['A', 'B', 'C', 'D', 'E']
            data = []
            for s in fmt['train']:
                data.append({
                    'text': s['text'],
                    'oracle_option': letters[s['label']],
                })
            n_train = len(data)
            for s in fmt['test'][:500]:
                data.append({
                    'text': s['text'],
                    'oracle_option': letters[s['label']],
                })
            train_idx = list(range(n_train))
            eval_idx = list(range(n_train, len(data)))
        else:
            print(f'  Unknown task {task} — skipping', flush=True)
            continue

        print(f'  {task}: {len(train_idx)} train, {len(eval_idx)} eval', flush=True)

        tag = f'peft_{args.method}_{model_short}_{task}_r{args.rank}_{args.total_steps//1000}k_s{args.seed}'
        try:
            train_and_eval_peft(
                base_model, tokenizer, data, train_idx, eval_idx,
                method=args.method, rank=args.rank,
                total_steps=args.total_steps, seed=args.seed,
                grad_accum=args.grad_accum, tag=tag,
                results_dir=args.results_dir, task_name=task,
                lr=args.lr, lora_plus_ratio=args.lora_plus_ratio,
            )
        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache()

    print(f'\n=== Done in {time.time()-overall_t0:.0f}s ===', flush=True)


if __name__ == '__main__':
    main()
