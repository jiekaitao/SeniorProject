"""
Thought Token Decoder — what is the controller actually thinking?

Train a controller, save it, then decode each thought token to its
nearest vocab words. Shows what the controller "writes" at each round.

This produces tables like:
  Round 1:
    Slot 0: "wall, north, →, ##, S"
    Slot 1: "path, ., open, E"
    ...
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/thought_decoder'
CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


class LowrankDeliberation(RecurrentDeliberation):
    def __init__(self, frozen_llm, rank=64, **kwargs):
        super().__init__(frozen_llm, **kwargs)
        d_state = kwargs.get('d_state', 512)
        self.to_lowrank = nn.Linear(d_state, rank, bias=False)
        self.U = nn.Parameter(torch.randn(rank, self.d_model) * 0.02)
        nn.init.normal_(self.to_lowrank.weight, std=0.01)

    def latent_to_thought_embs(self, z):
        E = self.frozen_llm.model.embed_tokens.weight
        logits = self.to_vocab_logits(z)
        vals, idx = logits.topk(self.topk_vocab, dim=-1)
        probs = F.softmax(vals, dim=-1)
        chosen_embs = E[idx]
        vocab_part = (probs.unsqueeze(-1) * chosen_embs).sum(dim=-2)
        lowrank_part = self.to_lowrank(z) @ self.U
        return vocab_part + 0.12 * lowrank_part

    def get_thought_decoding(self, z, tokenizer, top_k=10):
        """Decode each thought slot to its top-K vocabulary words."""
        E = self.frozen_llm.model.embed_tokens.weight
        logits = self.to_vocab_logits(z)  # (1, n_slots, V)

        decodings = []
        for slot in range(z.shape[1]):
            slot_logits = logits[0, slot]  # (V,)
            vals, idx = slot_logits.topk(top_k)
            probs = F.softmax(vals, dim=-1).cpu().tolist()
            tokens = [tokenizer.decode([i.item()]).strip() for i in idx]
            decodings.append(list(zip(tokens, probs)))
        return decodings

    def get_lowrank_nearest(self, z, tokenizer, top_k=10):
        """Decode lowrank residual to nearest vocab words."""
        E = self.frozen_llm.model.embed_tokens.weight  # (V, d_model)
        lowrank_part = self.to_lowrank(z) @ self.U  # (1, n_slots, d_model)

        decodings = []
        for slot in range(z.shape[1]):
            res = lowrank_part[0, slot].float()  # (d_model,)
            res_norm = res / (res.norm() + 1e-8)
            E_norm = E.float() / (E.float().norm(dim=-1, keepdim=True) + 1e-8)
            sims = E_norm @ res_norm
            top_vals, top_idx = sims.topk(top_k)
            tokens = [tokenizer.decode([i.item()]).strip() for i in top_idx]
            decodings.append(list(zip(tokens, top_vals.cpu().tolist())))
        return decodings


def get_choice_token_ids(tokenizer):
    ids = []
    for c in ['A', 'B', 'C', 'D']:
        toks = tokenizer.encode(f" {c}", add_special_tokens=False)
        ids.append(toks[0])
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_rounds', type=int, default=3)
    parser.add_argument('--n_slots', type=int, default=8)
    parser.add_argument('--decode_samples', type=int, default=5,
                        help='Number of mazes to decode')
    args = parser.parse_args()

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
    choice_ids_tensor = torch.tensor(choice_ids, device=device)
    lm_model = base_model.model

    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze_data = [s for s in ds if s['id'].startswith('mazenav')]
    random.seed(0)
    indices = list(range(len(maze_data)))
    random.shuffle(indices)
    train_idx, eval_idx = indices[:1000], indices[1000:]

    # Train controller
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    controller = LowrankDeliberation(
        frozen_llm=base_model, rank=64,
        d_state=512, n_slots=args.n_slots, tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    print(f'Training controller ({controller.count_trainable():,} params)...', flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05
    )
    warmup = 200
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (args.steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    t0 = time.time()
    losses_hist = []
    optimizer.zero_grad(set_to_none=True)

    for step in range(args.steps):
        sample = maze_data[train_idx[step % len(train_idx)]]
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

        all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=args.n_rounds)
        total_loss, loss_parts = controller.compute_loss(all_cl, all_v, label_tensor)
        total_loss = total_loss / 8
        total_loss.backward()

        if (step + 1) % 8 == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in controller.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        losses_hist.append(loss_parts['final_ce'])

        if (step + 1) % 500 == 0:
            avg = sum(losses_hist[-500:]) / len(losses_hist[-500:])
            print(f'  step {step+1} | ce={avg:.4f} | {time.time()-t0:.0f}s', flush=True)

    # Save controller
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(RESULTS_DIR, f'controller_seed{args.seed}.pt')
    torch.save({k: v for k, v in controller.state_dict().items() if 'frozen_llm' not in k}, ckpt_path)
    print(f'Saved controller to {ckpt_path}', flush=True)

    # ===========================
    # DECODE THOUGHT TOKENS
    # ===========================
    print(f'\n{"="*70}', flush=True)
    print(f'  THOUGHT TOKEN DECODING — what is the controller thinking?', flush=True)
    print(f'{"="*70}', flush=True)

    controller.eval()
    decoded_results = []

    for sample_i in range(args.decode_samples):
        idx = eval_idx[sample_i]
        sample = maze_data[idx]
        text = sample['text'][:1500]
        oracle = sample['oracle_option'].strip().upper()
        answer_label_val = CHOICE_MAP.get(oracle[0], 0)

        print(f'\n--- Sample {sample_i+1}: oracle={oracle} ---', flush=True)
        print(f'Maze (first 5 lines):', flush=True)
        for line in text.split('\n')[:8]:
            print(f'  {line}', flush=True)

        prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900,
                               add_special_tokens=True).to(device)
        answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                               add_special_tokens=False).to(device)

        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
            answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])

            # Run rounds and capture z at each step
            B = prompt_emb.shape[0]
            z = controller.z0.expand(B, -1, -1).clone()

            sample_decoding = {
                'sample_idx': sample_i,
                'oracle': oracle,
                'rounds': []
            }

            for r in range(args.n_rounds):
                # Decode the current state's vocab logits (top words by logit)
                vocab_decoding = controller.get_thought_decoding(z, tokenizer, top_k=10)
                lowrank_decoding = controller.get_lowrank_nearest(z, tokenizer, top_k=10)

                round_data = {
                    'round': r + 1,
                    'vocab_top10': vocab_decoding,
                    'lowrank_nearest': lowrank_decoding,
                }

                # Print formatted
                print(f'\n  ROUND {r+1}:', flush=True)
                for slot_i, (vocab_d, lowrank_d) in enumerate(zip(vocab_decoding, lowrank_decoding)):
                    vocab_str = ', '.join(f'"{w}"({p:.2f})' for w, p in vocab_d[:5])
                    lowrank_str = ', '.join(f'"{w}"' for w, _ in lowrank_d[:5])
                    print(f'    Slot {slot_i}: vocab=[{vocab_str}]', flush=True)
                    print(f'             lowrank=[{lowrank_str}]', flush=True)

                sample_decoding['rounds'].append(round_data)

                # Run forward to get next z
                thought_emb = controller.latent_to_thought_embs(z).to(prompt_emb.dtype)
                logits, think_h, tapped_pools = controller.forward_frozen_round(
                    prompt_emb, thought_emb, answer_emb)
                ans_logits = logits[:, -1, choice_ids_tensor]

                if r == args.n_rounds - 1:
                    pred = ans_logits.argmax(dim=-1).item()
                    pred_letter = ['A', 'B', 'C', 'D'][pred]
                    print(f'\n  Final prediction: {pred_letter} (oracle: {oracle}) — {"CORRECT" if pred == answer_label_val else "WRONG"}', flush=True)
                    sample_decoding['prediction'] = pred_letter
                    sample_decoding['correct'] = pred == answer_label_val
                    break

                feat = controller.build_features(think_h, tapped_pools, ans_logits)
                delta = controller.read_proj(feat).view(B, controller.n_slots, -1)
                z = controller.state_norm(z + controller.state_gate * delta)

            decoded_results.append(sample_decoding)

    # Save decoded results
    out_path = os.path.join(RESULTS_DIR, f'decoded_thoughts_seed{args.seed}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'seed': args.seed,
            'n_rounds': args.n_rounds,
            'n_slots': args.n_slots,
            'samples': decoded_results,
        }, f, indent=2)
    print(f'\nSaved decodings to {out_path}', flush=True)

    # Quick eval
    print(f'\n=== Eval ({len(eval_idx)} samples) ===', flush=True)
    correct = 0
    for idx in eval_idx:
        sample = maze_data[idx]
        text = sample['text'][:1500]
        oracle = sample['oracle_option'].strip().upper()
        answer_label_val = CHOICE_MAP.get(oracle[0], 0)
        prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
        answer_enc = tokenizer("\nAnswer:", return_tensors='pt', add_special_tokens=False).to(device)
        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
            answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
            all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=args.n_rounds)
            pred = all_cl[-1].argmax(dim=-1).item()
        if pred == answer_label_val:
            correct += 1
    acc = correct / len(eval_idx)
    print(f'  Accuracy: {acc:.4f} ({correct}/{len(eval_idx)})', flush=True)


if __name__ == '__main__':
    main()
