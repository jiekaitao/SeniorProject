"""
Hybrid Thought Writer — GPT-5.4's recommendation.

Current: thoughts = top-K sparse vocab superposition only
Proposed: thoughts = vocab_superposition + low_rank_residual

m = E^T * alpha + sigmoid(gate) * U * beta

Tests:
  vocab_only: current approach (control)
  hybrid_r32: vocab + rank-32 residual
  hybrid_r64: vocab + rank-64 residual
  lowrank_only: pure low-rank (no vocab constraint)
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation, RMSNorm

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/hybrid_writer'
CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


class HybridWriterDeliberation(RecurrentDeliberation):
    """Deliberation with hybrid vocab+lowrank thought writer."""
    def __init__(self, frozen_llm, writer_mode='vocab_only', rank=64, **kwargs):
        super().__init__(frozen_llm, **kwargs)
        self.writer_mode = writer_mode

        if writer_mode in ('hybrid', 'lowrank_only') or writer_mode.startswith('hybrid_r'):
            if writer_mode.startswith('hybrid_r'):
                rank = int(writer_mode.split('_r')[1])
            d_state = kwargs.get('d_state', 512)
            self.to_lowrank = nn.Linear(d_state, rank, bias=False)
            self.U = nn.Parameter(torch.randn(rank, self.d_model) * 0.02)
            self.writer_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2)≈0.12
            nn.init.normal_(self.to_lowrank.weight, std=0.01)

    def latent_to_thought_embs(self, z):
        E = self.frozen_llm.model.embed_tokens.weight
        logits = self.to_vocab_logits(z)
        vals, idx = logits.topk(self.topk_vocab, dim=-1)
        probs = F.softmax(vals, dim=-1)
        chosen_embs = E[idx]
        vocab_part = (probs.unsqueeze(-1) * chosen_embs).sum(dim=-2)

        if self.writer_mode == 'vocab_only':
            return vocab_part

        elif self.writer_mode.startswith('hybrid') or self.writer_mode == 'hybrid':
            lowrank_part = self.to_lowrank(z) @ self.U
            gate = torch.sigmoid(self.writer_gate)
            return vocab_part + gate * lowrank_part

        elif self.writer_mode == 'lowrank_only':
            lowrank_part = self.to_lowrank(z) @ self.U
            return lowrank_part

        return vocab_part


def load_data(task='mazenav'):
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    data = [s for s in ds if s['id'].startswith(task)]
    print(f'Loaded {len(data)} {task} samples', flush=True)
    return data


def get_choice_token_ids(tokenizer):
    ids = []
    for c in ['A', 'B', 'C', 'D']:
        toks = tokenizer.encode(f" {c}", add_special_tokens=False)
        ids.append(toks[0])
    return ids


def run_experiment(writer_mode, task, seed, total_steps, n_rounds,
                   tokenizer, base_model, data, train_idx, eval_idx, choice_ids):
    tag = f'hybrid_{writer_mode}_{task}_r{n_rounds}_seed{seed}'
    random.seed(seed)
    torch.manual_seed(seed)

    print(f'\n{"="*60}', flush=True)
    print(f'  Writer: {writer_mode} | Task: {task} | Seed: {seed}', flush=True)
    print(f'{"="*60}', flush=True)

    controller = HybridWriterDeliberation(
        frozen_llm=base_model, writer_mode=writer_mode,
        d_state=512, n_slots=8, tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    print(f'  Params: {controller.count_trainable():,}', flush=True)
    if hasattr(controller, 'writer_gate'):
        print(f'  Writer gate init: {torch.sigmoid(controller.writer_gate).item():.3f}', flush=True)

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
        sample = data[train_idx[step % len(train_idx)]]
        text = sample['text'][:1500]
        oracle = sample['oracle_option'].strip().upper()
        answer_label = CHOICE_MAP.get(oracle[0], 0)

        prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900,
                               add_special_tokens=True).to(device)
        answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                               add_special_tokens=False).to(device)

        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
            answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])

        label_tensor = torch.tensor([answer_label], device=device, dtype=torch.long)

        all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=n_rounds)
        total_loss, loss_parts = controller.compute_loss(all_cl, all_v, label_tensor)

        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in controller.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses_hist.append(loss_parts['final_ce'])

        if (step + 1) % 200 == 0:
            avg = sum(losses_hist[-200:]) / len(losses_hist[-200:])
            gate_str = ""
            if hasattr(controller, 'writer_gate'):
                gate_str = f' | gate={torch.sigmoid(controller.writer_gate).item():.3f}'
            print(f'  step {step+1} | ce={avg:.4f}{gate_str} | {time.time()-t0:.0f}s', flush=True)

    # Eval
    print(f'\n  === Eval ({len(eval_idx)} samples) ===', flush=True)
    controller.eval()
    results = {}

    for eval_rounds in [0, n_rounds]:
        correct = 0
        n_eval = len(eval_idx)

        for idx in eval_idx:
            sample = data[idx]
            text = sample['text'][:1500]
            oracle = sample['oracle_option'].strip().upper()
            answer_label_val = CHOICE_MAP.get(oracle[0], 0)

            if eval_rounds == 0:
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
                    all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=eval_rounds)
                    pred = all_cl[-1].argmax(dim=-1).item()
                if pred == answer_label_val:
                    correct += 1

        acc = correct / n_eval
        label = 'baseline' if eval_rounds == 0 else f'rounds={eval_rounds}'
        results[label] = {'accuracy': acc, 'correct': correct, 'total': n_eval}
        print(f'  {label}: {acc:.4f} ({correct}/{n_eval})', flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': tag, 'method': 'hybrid_writer_deliberation',
        'writer_mode': writer_mode, 'task': task,
        'n_rounds': n_rounds, 'seed': seed, 'total_steps': total_steps,
        'results': results,
        'final_loss': sum(losses_hist[-50:]) / max(len(losses_hist[-50:]), 1),
        'final_gate': torch.sigmoid(controller.writer_gate).item() if hasattr(controller, 'writer_gate') else None,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del controller, optimizer, scheduler
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', type=str, default='vocab_only,hybrid_r64')
    parser.add_argument('--task', type=str, default='mazenav')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seeds', type=str, default='42,7')
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()

    modes = args.modes.split(',')
    seeds = [int(x) for x in args.seeds.split(',')]

    print('Loading model...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    print('Model loaded.', flush=True)

    choice_ids = get_choice_token_ids(tokenizer)
    data = load_data(args.task)

    random.seed(0)
    indices = list(range(len(data)))
    random.shuffle(indices)
    split = min(1000, len(data) * 2 // 3)
    train_idx = indices[:split]
    eval_idx = indices[split:]

    for mode in modes:
        for seed in seeds:
            run_experiment(mode, args.task, seed, args.steps, args.rounds,
                          tokenizer, base_model, data, train_idx, eval_idx, choice_ids)

    print('\n=== All hybrid writer experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
