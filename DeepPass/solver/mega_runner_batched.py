"""
Batched mega runner - uses batch size 16 instead of 1 for actual GPU utilization.

Pads prompts to the longest in batch, uses attention masks properly.
Should push GPU util from ~2% to 50%+.
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation, RMSNorm
from eval_deliberation_creative import CHOICE_MAP

device = torch.device('cuda')


class BatchedMidLayerController(nn.Module):
    """
    Batched deliberation controller with mid-layer injection.
    Uses lowrank writer, supports batch > 1 via padding.
    """
    def __init__(self, frozen_llm, inject_layer=12, d_state=512, n_slots=8,
                 tapped_layers=(8, 16, 24), topk_vocab=64, rank=64):
        super().__init__()
        self.frozen_llm = frozen_llm
        self.inject_layer = inject_layer
        self.tapped_layers = set(tapped_layers)
        self.n_slots = n_slots
        self.topk_vocab = topk_vocab
        self.n_tap = len(tapped_layers)

        cfg = frozen_llm.config
        if hasattr(cfg, 'text_config'):
            self.d_model = cfg.text_config.hidden_size
            vocab_size = cfg.text_config.vocab_size
        else:
            self.d_model = cfg.hidden_size
            vocab_size = cfg.vocab_size

        for p in frozen_llm.parameters():
            p.requires_grad = False

        # Controller state
        self.z0 = nn.Parameter(torch.randn(1, n_slots, d_state) * 0.02)

        # Read features
        read_dim = self.n_tap * self.d_model + n_slots * self.d_model + 4 + 2
        self.read_proj = nn.Sequential(
            nn.Linear(read_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, n_slots * d_state),
        )
        self.state_norm = RMSNorm(d_state)
        self.state_gate = nn.Parameter(torch.tensor(0.1))

        # Vocab writer + lowrank residual
        self.to_vocab_logits = nn.Linear(d_state, vocab_size, bias=False)
        nn.init.normal_(self.to_vocab_logits.weight, std=0.01)
        self.to_lowrank = nn.Linear(d_state, rank, bias=False)
        self.U = nn.Parameter(torch.randn(rank, self.d_model) * 0.02)
        nn.init.normal_(self.to_lowrank.weight, std=0.01)

        # Verifier
        self.verifier = nn.Sequential(
            nn.Linear(self.n_tap * self.d_model + n_slots * self.d_model + 4, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )

    def latent_to_thought_embs(self, z):
        E = self.frozen_llm.model.embed_tokens.weight
        logits = self.to_vocab_logits(z)
        vals, idx = logits.topk(self.topk_vocab, dim=-1)
        probs = F.softmax(vals, dim=-1)
        chosen_embs = E[idx]
        vocab_part = (probs.unsqueeze(-1) * chosen_embs).sum(dim=-2)
        lowrank_part = self.to_lowrank(z) @ self.U
        return vocab_part + 0.12 * lowrank_part

    def forward_frozen_round(self, prompt_emb, thought_emb, answer_emb,
                             attention_mask):
        """Batched forward with mid-layer thought injection."""
        lm_model = self.frozen_llm.model
        B = prompt_emb.shape[0]
        p_len = prompt_emb.shape[1]
        a_len = answer_emb.shape[1]
        t_len = thought_emb.shape[1]

        # Combined sequence without thoughts initially
        dec_input = torch.cat([prompt_emb, answer_emb], dim=1)
        T = dec_input.shape[1]

        # Attention mask: prompt mask + answer mask (answer always unmasked)
        answer_mask = torch.ones(B, a_len, device=device, dtype=attention_mask.dtype)
        full_mask = torch.cat([attention_mask, answer_mask], dim=1)

        pos_ids = full_mask.cumsum(dim=-1) - 1
        pos_ids = pos_ids.clamp(min=0)
        pos_emb = lm_model.rotary_emb(dec_input, pos_ids)

        h = dec_input
        tapped_pools = []

        for i, layer in enumerate(lm_model.layers):
            if i == self.inject_layer:
                # Insert thought tokens between prompt and answer
                h_prompt = h[:, :p_len]
                h_answer = h[:, p_len:]
                h = torch.cat([h_prompt, thought_emb.to(h.dtype), h_answer], dim=1)

                # Extend attention mask
                thought_mask = torch.ones(B, t_len, device=device, dtype=full_mask.dtype)
                full_mask = torch.cat([
                    full_mask[:, :p_len], thought_mask, full_mask[:, p_len:]
                ], dim=1)

                # Recompute position IDs
                pos_ids = full_mask.cumsum(dim=-1) - 1
                pos_ids = pos_ids.clamp(min=0)
                pos_emb = lm_model.rotary_emb(h, pos_ids)

            # Create 4D attention mask for causal + padding
            T_cur = h.shape[1]
            attn_4d = torch.zeros(B, 1, T_cur, T_cur, device=device, dtype=h.dtype)
            # Causal mask
            causal = torch.triu(torch.ones(T_cur, T_cur, device=device), diagonal=1).bool()
            # Padding mask
            pad_mask = (full_mask == 0).unsqueeze(1).unsqueeze(2).expand(B, 1, T_cur, T_cur)
            mask_combined = causal.unsqueeze(0).unsqueeze(0) | pad_mask
            attn_4d = attn_4d.masked_fill(mask_combined, torch.finfo(h.dtype).min)

            h = layer(h, attention_mask=attn_4d, position_embeddings=pos_emb)[0]
            if i in self.tapped_layers:
                # Mean pool with mask
                mask_expand = full_mask.unsqueeze(-1).to(h.dtype)
                pooled = (h * mask_expand).sum(dim=1) / mask_expand.sum(dim=1).clamp(min=1)
                tapped_pools.append(pooled)

        h = lm_model.norm(h)
        logits = self.frozen_llm.lm_head(h)

        # Think slots hidden states (after injection)
        if self.inject_layer < len(lm_model.layers):
            think_h = h[:, p_len:p_len+t_len]
        else:
            think_h = h[:, :self.n_slots]

        return logits, think_h, tapped_pools

    def forward(self, prompt_emb, answer_emb, choice_ids, attention_mask, rounds=3):
        B = prompt_emb.shape[0]
        z = self.z0.expand(B, -1, -1).clone()
        all_cl = []
        all_v = []

        for r in range(rounds):
            thought_emb = self.latent_to_thought_embs(z).to(prompt_emb.dtype)
            logits, think_h, tapped_pools = self.forward_frozen_round(
                prompt_emb, thought_emb, answer_emb, attention_mask)

            # Answer logits at LAST position
            ans_logits = logits[:, -1, choice_ids]
            all_cl.append(ans_logits)

            # Build features
            dtype = think_h.dtype
            probs = ans_logits.float().softmax(dim=-1)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
            top2 = probs.topk(2, dim=-1).values
            margin = top2[:, :1] - top2[:, 1:2]
            feat = torch.cat(
                [think_h.flatten(1)] + tapped_pools +
                [probs.to(dtype), entropy.to(dtype), margin.to(dtype)],
                dim=-1
            )

            verify_feat = torch.cat(
                [think_h.flatten(1)] + tapped_pools + [probs.to(dtype)], dim=-1
            )
            verify = self.verifier(verify_feat)
            all_v.append(verify)

            if r < rounds - 1:
                delta = self.read_proj(feat).view(B, self.n_slots, -1)
                z = self.state_norm(z + self.state_gate * delta)

        return all_cl, all_v

    def compute_loss(self, all_cl, all_v, labels, lambda_v=0.5, lambda_p=0.1, delta_p=0.1):
        final_ce = F.cross_entropy(all_cl[-1].float(), labels)
        verify_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for r in range(len(all_cl)):
            pred_correct = (all_cl[r].argmax(dim=-1) == labels).float()
            verify_loss = verify_loss + F.binary_cross_entropy_with_logits(
                all_v[r].float().squeeze(-1), pred_correct)
        verify_loss = verify_loss / len(all_cl)

        progress_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        if len(all_cl) > 1:
            first_ce = F.cross_entropy(all_cl[0].float(), labels)
            progress_loss = F.relu(final_ce - first_ce + delta_p)

        total = final_ce + lambda_v * verify_loss + lambda_p * progress_loss
        return total, {'final_ce': final_ce.item()}

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_choice_tokens(tokenizer):
    ids = []
    for c in ['A', 'B', 'C', 'D']:
        toks = tokenizer.encode(f" {c}", add_special_tokens=False)
        ids.append(toks[0])
    return ids


def collate_batch(samples, tokenizer, lm_model, answer_emb, max_len=1900):
    """Collate a list of samples into a padded batch."""
    texts = [s['text'][:1500] for s in samples]
    labels = [CHOICE_MAP.get(s['oracle_option'].strip().upper()[0], 0) for s in samples]

    # Tokenize with padding
    encoded = tokenizer(
        texts, return_tensors='pt', truncation=True, max_length=max_len,
        padding=True, add_special_tokens=True
    ).to(device)

    with torch.no_grad():
        prompt_emb = lm_model.embed_tokens(encoded['input_ids'])

    # Broadcast answer_emb across batch
    B = prompt_emb.shape[0]
    answer_emb_batch = answer_emb.expand(B, -1, -1).contiguous()

    label_t = torch.tensor(labels, device=device, dtype=torch.long)
    return prompt_emb, answer_emb_batch, encoded['attention_mask'], label_t


def train_and_eval(base_model, tokenizer, lm_model, choice_ids_t,
                   answer_emb_single, data, train_idx, eval_idx,
                   inject_layer, n_rounds, total_steps, seed,
                   batch_size, tag, results_dir):
    random.seed(seed)
    torch.manual_seed(seed)

    controller = BatchedMidLayerController(
        frozen_llm=base_model, inject_layer=inject_layer,
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

    for step in range(total_steps):
        # Sample batch
        batch_indices = [train_idx[(step * batch_size + i) % len(train_idx)]
                        for i in range(batch_size)]
        batch_samples = [data[i] for i in batch_indices]

        prompt_emb, answer_emb_b, attn_mask, label_t = collate_batch(
            batch_samples, tokenizer, lm_model, answer_emb_single)

        all_cl, all_v = controller(prompt_emb, answer_emb_b, choice_ids_t,
                                   attn_mask, rounds=n_rounds)
        loss, lp = controller.compute_loss(all_cl, all_v, label_t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in controller.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(lp['final_ce'])
        if (step + 1) % 100 == 0:
            avg = sum(losses[-100:]) / len(losses[-100:])
            print(f'  step {step+1}/{total_steps} | ce={avg:.4f} | {time.time()-t0:.0f}s', flush=True)

    # Eval
    controller.eval()
    results = {}
    eval_batch_size = min(batch_size, 16)

    for er in [3, 5, 8]:
        if er > n_rounds and er > 3:
            if n_rounds < 3:
                continue
        correct = 0
        total = 0
        # Batched eval
        for i in range(0, len(eval_idx), eval_batch_size):
            batch_ids = eval_idx[i:i+eval_batch_size]
            batch_samples = [data[j] for j in batch_ids]

            prompt_emb, answer_emb_b, attn_mask, label_t = collate_batch(
                batch_samples, tokenizer, lm_model, answer_emb_single)

            with torch.no_grad():
                all_cl, _ = controller(prompt_emb, answer_emb_b, choice_ids_t,
                                       attn_mask, rounds=er)
                pred = all_cl[-1].argmax(dim=-1)
                correct += (pred == label_t).sum().item()
                total += label_t.shape[0]

        acc = correct / total
        results[f'rounds={er}'] = {'accuracy': acc, 'correct': correct, 'total': total}
        print(f'  rounds={er}: {acc:.4f}', flush=True)

    os.makedirs(results_dir, exist_ok=True)
    result_data = {
        'tag': tag, 'inject_layer': inject_layer, 'n_rounds': n_rounds,
        'total_steps': total_steps, 'seed': seed, 'batch_size': batch_size,
        'results': results,
    }
    with open(os.path.join(results_dir, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del controller, optimizer, scheduler
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/full/Llama-3.1-8B')
    parser.add_argument('--configs', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='results/data/mega')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Use left padding for decoder-only causal models
    tokenizer.padding_side = 'left'

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    lm_model = base_model.model
    print(f'Model loaded.', flush=True)

    choice_ids = get_choice_tokens(tokenizer)
    choice_ids_t = torch.tensor(choice_ids, device=device)

    # Pre-compute answer prefix embedding (same for all)
    answer_enc = tokenizer("\nAnswer:", return_tensors='pt', add_special_tokens=False).to(device)
    with torch.no_grad():
        answer_emb_single = lm_model.embed_tokens(answer_enc['input_ids'])

    # Load all tasks
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

    with open(args.configs) as f:
        configs = json.load(f)

    print(f'Running {len(configs)} experiments with batch_size={args.batch_size}...', flush=True)
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
                answer_emb_single, data, train_idx, eval_idx,
                inject_layer=cfg['inject_layer'],
                n_rounds=cfg.get('n_rounds', 3),
                total_steps=cfg.get('total_steps', 3000) // args.batch_size,  # adjust for batch
                seed=cfg['seed'],
                batch_size=args.batch_size,
                tag=cfg['tag'] + '_batch' + str(args.batch_size),
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
