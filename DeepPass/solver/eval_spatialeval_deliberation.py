"""
Recurrent Deliberation Controller — Training on SpatialEval.

The thesis architecture: frozen LLM + learned recurrent control interface.
Controller reads hidden states, writes vocab-space thought tokens, iterates.

Configs to sweep:
  - rounds: 1, 2, 3
  - n_slots: 4, 8, 16
  - topk_vocab: 32, 64, 128
  - tapped_layers: (8,16,24) vs (4,8,12,16,20,24,28)
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/spatialeval'
CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def load_maze_nav():
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze = [s for s in ds if s['id'].startswith('mazenav')]
    print(f'Loaded {len(maze)} Maze-Nav samples', flush=True)
    return maze


def get_choice_token_ids(tokenizer):
    ids = []
    for c in ['A', 'B', 'C', 'D']:
        toks = tokenizer.encode(f" {c}", add_special_tokens=False)
        ids.append(toks[0])
    return ids


def run_experiment(n_rounds, n_slots, topk_vocab, tapped_layers, seed, total_steps,
                   tokenizer, base_model, maze_data, train_idx, eval_idx, choice_ids):
    tap_str = '_'.join(map(str, tapped_layers))
    tag = f'delib_r{n_rounds}_s{n_slots}_k{topk_vocab}_t{tap_str}_seed{seed}'
    random.seed(seed)
    torch.manual_seed(seed)
    print(f'\n{"="*60}', flush=True)
    print(f'  Deliberation: rounds={n_rounds}, slots={n_slots}, topk={topk_vocab}', flush=True)
    print(f'  tapped={tapped_layers}, seed={seed}', flush=True)
    print(f'{"="*60}', flush=True)

    controller = RecurrentDeliberation(
        frozen_llm=base_model,
        d_state=512,
        n_slots=n_slots,
        tapped_layers=tuple(tapped_layers),
        topk_vocab=topk_vocab,
    ).to(device=device, dtype=torch.bfloat16)

    n_params = controller.count_trainable()
    print(f'  Trainable params: {n_params:,} ({n_params/1e6:.1f}M)', flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05
    )
    warmup = 200
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    choice_ids_tensor = torch.tensor(choice_ids, device=device)
    lm_model = base_model.model
    t0 = time.time()
    losses_hist = []

    for step in range(total_steps):
        sample = maze_data[train_idx[step % len(train_idx)]]
        text = sample['text'][:1500]
        oracle = sample['oracle_option'].strip().upper()
        answer_label = CHOICE_MAP.get(oracle[0], 0)

        # Tokenize prompt and answer prefix separately
        prompt_text = text
        answer_prefix = "\nAnswer:"

        prompt_enc = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=1900,
                               add_special_tokens=True).to(device)
        answer_enc = tokenizer(answer_prefix, return_tensors='pt',
                               add_special_tokens=False).to(device)

        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])  # (1, P, 4096)
            answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])  # (1, A, 4096)

        label_tensor = torch.tensor([answer_label], device=device, dtype=torch.long)

        # Forward through deliberation controller
        all_choice_logits, all_verify = controller(
            prompt_emb, answer_emb, choice_ids_tensor, rounds=n_rounds
        )

        # Compute loss
        total_loss, loss_parts = controller.compute_loss(
            all_choice_logits, all_verify, label_tensor,
            lambda_v=0.5, lambda_p=0.1, delta_p=0.1
        )

        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in controller.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses_hist.append(loss_parts['final_ce'])

        if (step + 1) % 100 == 0:
            avg = sum(losses_hist[-100:]) / len(losses_hist[-100:])
            r1_acc = (all_choice_logits[0].argmax(dim=-1) == label_tensor).float().item()
            rN_acc = (all_choice_logits[-1].argmax(dim=-1) == label_tensor).float().item()
            v1 = torch.sigmoid(all_verify[0]).item()
            print(f'  step {step+1} | ce={avg:.4f} | r1_correct={r1_acc:.0f} rN_correct={rN_acc:.0f} '
                  f'| verify_r1={v1:.3f} | prog={loss_parts["progress_loss"]:.4f} '
                  f'| {time.time()-t0:.0f}s', flush=True)

    # ========== EVAL ==========
    print(f'\n  === Eval ({len(eval_idx)} samples) ===', flush=True)
    controller.eval()
    results = {}

    for eval_rounds in [0, 1, n_rounds]:
        correct = 0
        per_round_correct = [0] * max(eval_rounds, 1)
        verify_correct = 0
        n_eval = len(eval_idx)

        for idx in eval_idx:
            sample = maze_data[idx]
            text = sample['text'][:1500]
            oracle = sample['oracle_option'].strip().upper()
            answer_label_val = CHOICE_MAP.get(oracle[0], 0)

            if eval_rounds == 0:
                # Baseline: frozen model only
                prompt = text + "\nAnswer:"
                enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
                with torch.no_grad():
                    out = base_model.generate(enc['input_ids'], max_new_tokens=5, do_sample=False,
                                               pad_token_id=tokenizer.pad_token_id)
                    answer = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                if oracle in answer.upper()[:10]:
                    correct += 1
            else:
                prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900,
                                       add_special_tokens=True).to(device)
                answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                                       add_special_tokens=False).to(device)
                with torch.no_grad():
                    prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
                    answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
                    all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_tensor,
                                               rounds=eval_rounds)

                    # Check each round
                    for r in range(eval_rounds):
                        if all_cl[r].argmax(dim=-1).item() == answer_label_val:
                            per_round_correct[r] += 1

                    # Final round answer
                    pred = all_cl[-1].argmax(dim=-1).item()
                    if pred == answer_label_val:
                        correct += 1

                    # Verifier accuracy
                    v_pred = (torch.sigmoid(all_v[-1]) > 0.5).item()
                    v_truth = (pred == answer_label_val)
                    if v_pred == v_truth:
                        verify_correct += 1

        acc = correct / n_eval
        results[f'rounds={eval_rounds}'] = {
            'accuracy': acc, 'correct': correct, 'total': n_eval,
        }
        if eval_rounds > 0:
            results[f'rounds={eval_rounds}']['per_round'] = [c/n_eval for c in per_round_correct]
            results[f'rounds={eval_rounds}']['verifier_acc'] = verify_correct / n_eval

        if eval_rounds == 0:
            print(f'  baseline (no controller): {acc:.4f} ({correct}/{n_eval})', flush=True)
        else:
            per_r_str = ' '.join(f'r{i+1}={c/n_eval:.3f}' for i, c in enumerate(per_round_correct))
            print(f'  rounds={eval_rounds}: {acc:.4f} ({correct}/{n_eval}) | {per_r_str} | '
                  f'verify_acc={verify_correct/n_eval:.3f}', flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': tag, 'method': 'recurrent_deliberation',
        'n_rounds': n_rounds, 'n_slots': n_slots, 'topk_vocab': topk_vocab,
        'tapped_layers': tapped_layers,
        'seed': seed, 'total_steps': total_steps,
        'trainable_params': n_params,
        'final_loss': sum(losses_hist[-50:]) / max(len(losses_hist[-50:]), 1),
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'spatialeval_{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: spatialeval_{tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del controller, optimizer, scheduler
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=str, default='2')
    parser.add_argument('--slots', type=str, default='8')
    parser.add_argument('--topk', type=int, default=64)
    parser.add_argument('--tapped', type=str, default='8,16,24')
    parser.add_argument('--seeds', type=str, default='42,7')
    parser.add_argument('--steps', type=int, default=2000)
    args = parser.parse_args()

    rounds_list = [int(x) for x in args.rounds.split(',')]
    slots_list = [int(x) for x in args.slots.split(',')]
    tapped = [int(x) for x in args.tapped.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

    print('Loading model...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False

    choice_ids = get_choice_token_ids(tokenizer)
    print(f'Choice token IDs: {dict(zip(["A","B","C","D"], choice_ids))}', flush=True)

    maze_data = load_maze_nav()
    random.seed(0)
    indices = list(range(len(maze_data)))
    random.shuffle(indices)
    train_idx = indices[:1000]
    eval_idx = indices[1000:]

    for n_rounds in rounds_list:
        for n_slots in slots_list:
            for seed in seeds:
                run_experiment(n_rounds, n_slots, args.topk, tapped, seed,
                              args.steps, tokenizer, base_model, maze_data,
                              train_idx, eval_idx, choice_ids)

    print('\n=== All deliberation experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
