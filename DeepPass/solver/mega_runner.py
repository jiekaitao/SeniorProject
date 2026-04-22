"""
Mega runner: loads model ONCE, runs many experiments.
This fixes the GPU underutilization from spawning many subprocesses.
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation
from eval_deliberation_creative import MidLayerDeliberation, CHOICE_MAP

device = torch.device('cuda')


def get_choice_tokens(tokenizer):
    ids = []
    for c in ['A', 'B', 'C', 'D']:
        toks = tokenizer.encode(f" {c}", add_special_tokens=False)
        ids.append(toks[0])
    return ids


def train_and_eval(base_model, tokenizer, lm_model, choice_ids_t,
                   data, train_idx, eval_idx, inject_layer, n_rounds,
                   total_steps, seed, grad_accum, tag, results_dir):
    """Run a single training + eval experiment."""
    random.seed(seed)
    torch.manual_seed(seed)

    controller = MidLayerDeliberation(
        frozen_llm=base_model, inject_layer=inject_layer, rank=64,
        d_state=512, n_slots=8, tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05
    )
    warmup = 200
    def lr_sched(s):
        if s < warmup: return s / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    t0 = time.time()
    losses = []
    optimizer.zero_grad(set_to_none=True)

    for step in range(total_steps):
        sample = data[train_idx[step % len(train_idx)]]
        text = sample['text'][:1500]
        oracle = sample['oracle_option'].strip().upper()
        answer_label = CHOICE_MAP.get(oracle[0], 0)
        prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
        answer_enc = tokenizer("\nAnswer:", return_tensors='pt', add_special_tokens=False).to(device)
        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
            answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
        label_t = torch.tensor([answer_label], device=device, dtype=torch.long)
        all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_t, rounds=n_rounds)
        loss, lp = controller.compute_loss(all_cl, all_v, label_t)
        loss = loss / grad_accum
        loss.backward()
        if (step+1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in controller.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(lp['final_ce'])
        if (step+1) % 1000 == 0:
            avg = sum(losses[-1000:])/1000
            print(f'  step {step+1}/{total_steps} | ce={avg:.4f} | {time.time()-t0:.0f}s', flush=True)

    controller.eval()
    results = {}

    # K-scaling eval (multi-round)
    for er in [3, 5, 8]:
        if er > n_rounds and er > 3:
            # Only eval higher rounds if trained with enough rounds
            if n_rounds < 3:
                continue
        correct = 0
        for idx in eval_idx:
            sample = data[idx]
            text = sample['text'][:1500]
            oracle = sample['oracle_option'].strip().upper()
            al = CHOICE_MAP.get(oracle[0], 0)
            prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
            answer_enc = tokenizer("\nAnswer:", return_tensors='pt', add_special_tokens=False).to(device)
            with torch.no_grad():
                prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
                answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
                all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_t, rounds=er)
                if all_cl[-1].argmax(dim=-1).item() == al: correct += 1
        acc = correct / len(eval_idx)
        results[f'rounds={er}'] = {'accuracy': acc, 'correct': correct, 'total': len(eval_idx)}
        print(f'  rounds={er}: {acc:.4f}', flush=True)

    os.makedirs(results_dir, exist_ok=True)
    result_data = {
        'tag': tag, 'inject_layer': inject_layer, 'n_rounds': n_rounds,
        'total_steps': total_steps, 'seed': seed, 'grad_accum': grad_accum,
        'results': results,
    }
    with open(os.path.join(results_dir, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del controller, optimizer, scheduler
    torch.cuda.empty_cache()
    return result_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/full/Llama-3.1-8B')
    parser.add_argument('--configs', type=str, required=True,
                        help='JSON file with list of experiment configs')
    parser.add_argument('--results_dir', type=str, default='results/data/creative')
    args = parser.parse_args()

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    lm_model = base_model.model
    print(f'Model loaded.', flush=True)

    choice_ids = get_choice_tokens(tokenizer)
    choice_ids_t = torch.tensor(choice_ids, device=device)

    # Pre-load all needed datasets
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')

    task_data = {}
    task_train_idx = {}
    task_eval_idx = {}
    for task in ['mazenav', 'spatialmap', 'spatialgrid', 'spatialreal']:
        task_data[task] = [s for s in ds if s['id'].startswith(task)]
        random.seed(0)
        indices = list(range(len(task_data[task])))
        random.shuffle(indices)
        split = min(1000, len(indices) * 2 // 3)
        task_train_idx[task] = indices[:split]
        task_eval_idx[task] = indices[split:]
        print(f'  {task}: {len(task_train_idx[task])} train, {len(task_eval_idx[task])} eval', flush=True)

    # Load configs
    with open(args.configs) as f:
        configs = json.load(f)

    print(f'Running {len(configs)} experiments...', flush=True)
    overall_t0 = time.time()

    for i, cfg in enumerate(configs):
        print(f'\n{"="*70}', flush=True)
        print(f'[{i+1}/{len(configs)}] {cfg.get("tag", "?")}', flush=True)
        print(f'{"="*70}', flush=True)

        task = cfg.get('task', 'mazenav')
        data = task_data[task]
        train_idx = task_train_idx[task]
        eval_idx = task_eval_idx[task]

        try:
            train_and_eval(
                base_model, tokenizer, lm_model, choice_ids_t,
                data, train_idx, eval_idx,
                inject_layer=cfg['inject_layer'],
                n_rounds=cfg.get('n_rounds', 3),
                total_steps=cfg.get('total_steps', 3000),
                seed=cfg['seed'],
                grad_accum=cfg.get('grad_accum', 8),
                tag=cfg['tag'],
                results_dir=args.results_dir,
            )
        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()

    print(f'\n=== All {len(configs)} experiments done in {time.time()-overall_t0:.0f}s ===', flush=True)


if __name__ == '__main__':
    main()
