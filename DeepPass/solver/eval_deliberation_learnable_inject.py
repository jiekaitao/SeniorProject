"""
Learnable Injection Layer Experiments.

Experiment 1: MultiLayer - inject at every layer with learned per-layer gate
Experiment 2: Router - small classifier picks injection layer per input
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation, RMSNorm

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/learnable_inject'
CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


class MultiLayerDeliberation(RecurrentDeliberation):
    """
    Inject thought tokens at EVERY layer, with a learned per-layer gate.
    The model figures out which layers benefit from injection.
    """
    def __init__(self, frozen_llm, rank=64, **kwargs):
        super().__init__(frozen_llm, **kwargs)
        d_state = kwargs.get('d_state', 512)
        n_llm_layers = len(frozen_llm.model.layers)

        # Lowrank writer
        self.to_lowrank = nn.Linear(d_state, rank, bias=False)
        self.U = nn.Parameter(torch.randn(rank, self.d_model) * 0.02)
        nn.init.normal_(self.to_lowrank.weight, std=0.01)

        # Per-layer injection gates (start at -2 = sigmoid 0.12)
        self.layer_gates = nn.Parameter(torch.full((n_llm_layers,), -2.0))

    def latent_to_thought_embs(self, z):
        E = self.frozen_llm.model.embed_tokens.weight
        logits = self.to_vocab_logits(z)
        vals, idx = logits.topk(self.topk_vocab, dim=-1)
        probs = F.softmax(vals, dim=-1)
        chosen_embs = E[idx]
        vocab_part = (probs.unsqueeze(-1) * chosen_embs).sum(dim=-2)
        lowrank_part = self.to_lowrank(z) @ self.U
        return vocab_part + 0.12 * lowrank_part

    def forward_frozen_round(self, prompt_emb, thought_emb, answer_emb):
        """Inject thoughts at every layer with learned gate."""
        lm_model = self.frozen_llm.model

        # Use only [prompt | answer] as the active sequence
        # Thoughts are added as a residual at each layer
        dec_input = torch.cat([prompt_emb, answer_emb], dim=1)
        T = dec_input.shape[1]
        pos_ids = torch.arange(T, device=dec_input.device).unsqueeze(0)
        pos_emb = lm_model.rotary_emb(dec_input, pos_ids)

        # Pool thoughts to a single per-position residual
        # thought_emb shape: (B, n_slots, d_model) -> mean -> (B, 1, d_model)
        thought_pool = thought_emb.mean(dim=1, keepdim=True)  # (B, 1, d_model)
        # Broadcast to sequence length
        thought_residual = thought_pool.expand(-1, T, -1)  # (B, T, d_model)

        h = dec_input
        tapped_pools = []

        for i, layer in enumerate(lm_model.layers):
            # Inject thoughts as residual with learned gate
            gate = torch.sigmoid(self.layer_gates[i])
            h = h + gate * thought_residual

            h = layer(h, position_embeddings=pos_emb)

            if i in self.tapped_layers:
                tapped_pools.append(h.mean(dim=1))

        h = lm_model.norm(h)
        logits = self.frozen_llm.lm_head(h)

        # Use answer position hidden state as "think_h" placeholder
        think_h = h[:, :self.n_slots]
        return logits, think_h, tapped_pools


class RouterDeliberation(RecurrentDeliberation):
    """
    A router predicts which layer to inject at, based on the prompt.
    Uses gumbel-softmax for differentiable hard selection during training.
    """
    def __init__(self, frozen_llm, rank=64,
                 candidate_layers=(8, 12, 14, 16, 18, 20, 24), **kwargs):
        super().__init__(frozen_llm, **kwargs)
        d_state = kwargs.get('d_state', 512)
        self.candidate_layers = list(candidate_layers)

        # Lowrank writer
        self.to_lowrank = nn.Linear(d_state, rank, bias=False)
        self.U = nn.Parameter(torch.randn(rank, self.d_model) * 0.02)
        nn.init.normal_(self.to_lowrank.weight, std=0.01)

        # Router: takes pooled prompt embedding -> distribution over candidates
        self.router = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.GELU(),
            nn.Linear(256, len(self.candidate_layers))
        )
        # Routing temperature (anneal from soft to hard)
        self.tau = 1.0

    def latent_to_thought_embs(self, z):
        E = self.frozen_llm.model.embed_tokens.weight
        logits = self.to_vocab_logits(z)
        vals, idx = logits.topk(self.topk_vocab, dim=-1)
        probs = F.softmax(vals, dim=-1)
        chosen_embs = E[idx]
        vocab_part = (probs.unsqueeze(-1) * chosen_embs).sum(dim=-2)
        lowrank_part = self.to_lowrank(z) @ self.U
        return vocab_part + 0.12 * lowrank_part

    def forward_frozen_round(self, prompt_emb, thought_emb, answer_emb):
        """Router picks injection layer per input via gumbel-softmax."""
        lm_model = self.frozen_llm.model

        # Compute routing distribution from prompt
        prompt_summary = prompt_emb.mean(dim=1)  # (B, d_model)
        route_logits = self.router(prompt_summary)  # (B, n_candidates)

        # Hard gumbel-softmax during training, argmax during eval
        if self.training:
            route_weights = F.gumbel_softmax(route_logits, tau=self.tau, hard=True, dim=-1)
        else:
            idx = route_logits.argmax(dim=-1)
            route_weights = F.one_hot(idx, num_classes=len(self.candidate_layers)).float()

        # Run through layers, injecting at the routed layer
        # For batch=1 we can use the argmax directly
        dec_input = torch.cat([prompt_emb, answer_emb], dim=1)
        T = dec_input.shape[1]
        pos_ids = torch.arange(T, device=dec_input.device).unsqueeze(0)
        pos_emb = lm_model.rotary_emb(dec_input, pos_ids)

        h = dec_input
        tapped_pools = []
        injected = False

        for i, layer in enumerate(lm_model.layers):
            # Check if this layer is a candidate
            if i in self.candidate_layers:
                cand_idx = self.candidate_layers.index(i)
                # Get the routing weight for this layer
                w = route_weights[:, cand_idx].view(-1, 1, 1)  # (B, 1, 1)

                if w.sum() > 0.001:  # Inject thoughts here
                    t_len = thought_emb.shape[1]
                    # Insert thought tokens (weighted by route)
                    thought_inject = thought_emb.to(h.dtype) * w
                    h = torch.cat([
                        h[:, :prompt_emb.shape[1]],
                        thought_inject,
                        h[:, prompt_emb.shape[1]:]
                    ], dim=1)
                    T_new = h.shape[1]
                    pos_ids = torch.arange(T_new, device=h.device).unsqueeze(0)
                    pos_emb = lm_model.rotary_emb(h, pos_ids)
                    injected = True

            h = layer(h, position_embeddings=pos_emb)
            if i in self.tapped_layers:
                tapped_pools.append(h.mean(dim=1))

        h = lm_model.norm(h)
        logits = self.frozen_llm.lm_head(h)

        # Track think slots if injected
        if injected:
            # Find think positions (between prompt and answer)
            p_len = prompt_emb.shape[1]
            t_len = thought_emb.shape[1]
            think_h = h[:, p_len:p_len+t_len]
        else:
            think_h = h[:, :self.n_slots]

        return logits, think_h, tapped_pools


def get_choice_token_ids(tokenizer):
    ids = []
    for c in ['A', 'B', 'C', 'D']:
        toks = tokenizer.encode(f" {c}", add_special_tokens=False)
        ids.append(toks[0])
    return ids


def run_experiment(method, seed, total_steps, n_rounds, tokenizer, base_model,
                   maze_data, train_idx, eval_idx, choice_ids):
    tag = f'{method}_r{n_rounds}_seed{seed}'
    random.seed(seed)
    torch.manual_seed(seed)

    print(f'\n{"="*60}', flush=True)
    print(f'  {method.upper()}: rounds={n_rounds} | seed={seed}', flush=True)
    print(f'{"="*60}', flush=True)

    if method == 'multilayer':
        controller = MultiLayerDeliberation(
            frozen_llm=base_model, rank=64,
            d_state=512, n_slots=8, tapped_layers=(8, 16, 24), topk_vocab=64,
        ).to(device=device, dtype=torch.bfloat16)
    elif method == 'router':
        controller = RouterDeliberation(
            frozen_llm=base_model, rank=64,
            candidate_layers=(8, 12, 14, 16, 18, 20, 24),
            d_state=512, n_slots=8, tapped_layers=(8, 16, 24), topk_vocab=64,
        ).to(device=device, dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f'  Params: {controller.count_trainable():,}', flush=True)

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
    optimizer.zero_grad(set_to_none=True)

    for step in range(total_steps):
        # Anneal router temperature
        if method == 'router':
            controller.tau = max(0.5, 1.0 - (step / total_steps) * 0.5)

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

        all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=n_rounds)
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
            extra = ""
            if method == 'multilayer':
                gates = torch.sigmoid(controller.layer_gates).tolist()
                top3 = sorted(enumerate(gates), key=lambda x: -x[1])[:3]
                extra = f" | top gates: {top3}"
            elif method == 'router':
                # Show distribution of choices
                with torch.no_grad():
                    test_route = controller.router(prompt_emb.mean(dim=1))
                    pick = test_route.argmax(dim=-1).item()
                    extra = f" | pick={controller.candidate_layers[pick]} | tau={controller.tau:.2f}"
            print(f'  step {step+1} | ce={avg:.4f}{extra} | {time.time()-t0:.0f}s', flush=True)

    # Eval
    print(f'\n  === Eval ({len(eval_idx)} samples) ===', flush=True)
    controller.eval()
    correct = 0
    n_eval = len(eval_idx)
    layer_picks = {}  # for router

    for idx in eval_idx:
        sample = maze_data[idx]
        text = sample['text'][:1500]
        oracle = sample['oracle_option'].strip().upper()
        answer_label_val = CHOICE_MAP.get(oracle[0], 0)

        prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900,
                               add_special_tokens=True).to(device)
        answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                               add_special_tokens=False).to(device)
        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
            answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])

            if method == 'router':
                # Track which layer was picked
                summary = prompt_emb.mean(dim=1)
                route_logits = controller.router(summary)
                pick = controller.candidate_layers[route_logits.argmax(dim=-1).item()]
                layer_picks[pick] = layer_picks.get(pick, 0) + 1

            all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=n_rounds)
            pred = all_cl[-1].argmax(dim=-1).item()
        if pred == answer_label_val:
            correct += 1

    acc = correct / n_eval
    print(f'  FINAL {tag}: {acc:.4f} ({correct}/{n_eval})', flush=True)

    # Report learned configurations
    if method == 'multilayer':
        gates = torch.sigmoid(controller.layer_gates).tolist()
        sorted_gates = sorted(enumerate(gates), key=lambda x: -x[1])
        print(f'  Top 5 layer gates: {sorted_gates[:5]}', flush=True)
        print(f'  All gates: {[round(g, 3) for g in gates]}', flush=True)
    elif method == 'router':
        print(f'  Layer pick distribution: {layer_picks}', flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': tag, 'method': method,
        'n_rounds': n_rounds, 'seed': seed, 'total_steps': total_steps,
        'accuracy': acc, 'correct': correct, 'total': n_eval,
        'final_loss': sum(losses_hist[-50:]) / max(len(losses_hist[-50:]), 1),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if method == 'multilayer':
        result_data['layer_gates'] = [float(g) for g in gates]
    elif method == 'router':
        result_data['layer_picks'] = {str(k): v for k, v in layer_picks.items()}

    with open(os.path.join(RESULTS_DIR, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del controller, optimizer
    torch.cuda.empty_cache()
    return result_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='multilayer,router')
    parser.add_argument('--seeds', type=str, default='42,7')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()

    methods = args.methods.split(',')
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

    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze_data = [s for s in ds if s['id'].startswith('mazenav')]
    random.seed(0)
    indices = list(range(len(maze_data)))
    random.shuffle(indices)
    train_idx, eval_idx = indices[:1000], indices[1000:]

    for method in methods:
        for seed in seeds:
            run_experiment(method, seed, args.steps, args.rounds, tokenizer, base_model,
                          maze_data, train_idx, eval_idx, choice_ids)

    print('\n=== All learnable injection experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
