"""Thought-gated DoRA hybrid controller.

Combines the recurrent deliberation controller (good on SpatialGrid) with
a small DoRA branch (good on semantic tasks like WinoGrande/HellaSwag).
The controller decides per-instance and per-round how much of the DoRA
delta to apply at each wrapped site by emitting a gate value in [0, 1].

Architecture sketch per round::

    z_r --latent_to_thoughts--> thought_emb --inject@L12--> frozen LM fwd
     |                                                           ^
     |                                                           |
     |   gate_head(z_r or prev feat) -> alpha_per_site  ---------/
     |                                                           |
    v (verifier)  <--- features (tapped pools, think_h, logits) -+
     |
    z_{r+1} <- state_norm(z_r + state_gate * read_proj(feat))

Round 0 uses a small fixed alpha (0.05) so the very first forward behaves
close to a pure deliberation controller. Rounds 1..R-1 set alphas from
the gate_head applied to a gate input (default: flattened controller state
``z``, shape ``n_slots * d_state`` — kept small so the gate head adds
only ~0.05M params even with 60 sites).

The gate input can alternatively be the previous round's full ``feat``
(much larger), but that makes the gate head dominate the parameter budget.
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from mega_runner_benchmarks import (
    FlexMidLayerDeliberation, BENCHMARK_LOADERS, get_choice_tokens,
)
from gated_dora_layers import (
    wrap_model_with_gated_dora, set_gate_alphas, clear_gate_alphas,
    count_dora_params,
)

device = torch.device('cuda')
CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}


class HybridDoRAController(FlexMidLayerDeliberation):
    """FlexMidLayerDeliberation that also emits per-site gates for
    :class:`GatedDoRALinear` layers wrapped on the frozen LM.
    """

    def __init__(self, frozen_llm, sites, n_choices=4,
                 gate_hidden=128, initial_alpha=0.05, **kwargs):
        super().__init__(frozen_llm, n_choices=n_choices, **kwargs)
        self.sites = sites
        self.n_sites = len(sites)
        self.initial_alpha = float(initial_alpha)

        # Gate input: flattened controller state z (n_slots * d_state).
        d_state = kwargs.get('d_state', 512)
        gate_in = self.n_slots * d_state
        self.gate_head = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.n_sites),
        )
        # Bias the final layer so sigmoid starts near 0 — keeps the model
        # as a pure controller at init.
        with torch.no_grad():
            # target sigmoid(x) = initial_alpha  =>  x = logit(initial_alpha)
            eps = 1e-6
            init_bias = math.log(self.initial_alpha + eps) - math.log(1.0 - self.initial_alpha + eps)
            self.gate_head[-1].bias.fill_(init_bias)
            # Small weight so gates don't move wildly at first.
            nn.init.normal_(self.gate_head[-1].weight, std=0.01)

        self.last_alphas = None  # populated during forward, used in loss

    def _compute_alphas(self, z):
        """Compute per-site gate alphas from current controller state z.

        z: (B, n_slots, d_state)
        returns (B, n_sites) in [0, 1].
        """
        B = z.shape[0]
        gate_in = z.reshape(B, -1)
        # gate_head is fp32 by default; cast to z's dtype so downstream
        # sigmoid runs in bf16 consistently with the model.
        alpha_logits = self.gate_head(gate_in.to(next(self.gate_head.parameters()).dtype))
        alpha = torch.sigmoid(alpha_logits).to(z.dtype)
        return alpha

    def forward(self, prompt_emb, answer_emb, choice_ids, rounds=2):
        B = prompt_emb.shape[0]
        z = self.z0.expand(B, -1, -1).clone()
        nc = choice_ids.shape[0]

        all_choice_logits = []
        all_verify = []
        alphas_per_round = []

        for r in range(rounds):
            # Write: thought embeddings
            thought_emb = self.latent_to_thought_embs(z)

            # Compute & set gate alphas.
            # Round 0: use a small fixed alpha so the very first forward
            # is close to the pure controller (DoRA has little effect).
            if r == 0:
                alpha_per_site = torch.full(
                    (B, self.n_sites), self.initial_alpha,
                    device=z.device, dtype=z.dtype,
                )
            else:
                alpha_per_site = self._compute_alphas(z)
            alphas_per_round.append(alpha_per_site)
            set_gate_alphas(self.sites, alpha_per_site)

            try:
                logits, think_h, tapped_pools = self.forward_frozen_round(
                    prompt_emb, thought_emb.to(prompt_emb.dtype), answer_emb
                )
            finally:
                # Always clear so a stray standalone forward can't pick up
                # stale alphas (important for the baseline / eval paths).
                clear_gate_alphas(self.sites)

            ans_logits = logits[:, -1, choice_ids]
            all_choice_logits.append(ans_logits)

            # Build features (same as parent)
            dtype = think_h.dtype
            probs = ans_logits.float().softmax(dim=-1).to(dtype)
            entropy = -(probs.float() * probs.float().clamp_min(1e-8).log()).sum(
                dim=-1, keepdim=True).to(dtype)
            top2 = probs.float().topk(min(2, nc), dim=-1).values
            margin = (top2[:, :1] - top2[:, 1:2]).to(dtype) if top2.shape[-1] >= 2 else top2[:, :1].to(dtype)

            feat = torch.cat(
                [think_h.flatten(1)] + tapped_pools + [probs, entropy, margin],
                dim=-1,
            )
            verify_feat = torch.cat(
                [think_h.flatten(1)] + tapped_pools + [probs],
                dim=-1,
            )
            verify = self.verifier(verify_feat)
            all_verify.append(verify)

            if r < rounds - 1:
                delta = self.read_proj(feat).view(B, self.n_slots, -1)
                z = self.state_norm(z + self.state_gate * delta)

        # Stash alphas for the loss's sparsity penalty.
        self.last_alphas = torch.stack(alphas_per_round, dim=1)  # (B, R, n_sites)
        return all_choice_logits, all_verify

    def compute_loss(self, all_choice_logits, all_verify, answer_labels,
                     lambda_v=0.5, lambda_p=0.1, lambda_sparse=0.001):
        loss, parts = super().compute_loss(
            all_choice_logits, all_verify, answer_labels,
            lambda_v=lambda_v, lambda_p=lambda_p,
        )
        if self.last_alphas is not None:
            sparse_loss = self.last_alphas.float().abs().mean()
            loss = loss + lambda_sparse * sparse_loss
            parts['sparse_loss'] = sparse_loss.item()
            parts['alpha_mean'] = self.last_alphas.float().mean().item()
            parts['alpha_max'] = self.last_alphas.float().max().item()
        return loss, parts


# ---------------------------------------------------------------------------
# SpatialEval loader (mirrors mega_runner.py)
# ---------------------------------------------------------------------------

def load_spatialeval_task(task_name, seed_for_split=0):
    """Load a SpatialEval TQA task (spatialgrid / mazenav / spatialmap / spatialreal).

    Returns ``(data, train_idx, eval_idx, n_choices=4)``.
    """
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    task_data = [s for s in ds if s['id'].startswith(task_name)]
    random.seed(seed_for_split)
    indices = list(range(len(task_data)))
    random.shuffle(indices)
    split = min(1000, len(indices) * 2 // 3)
    train_idx = indices[:split]
    eval_idx = indices[split:]
    return task_data, train_idx, eval_idx, 4


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_and_eval_hybrid_benchmark(base_model, tokenizer, lm_model, sites,
                                      data, n_choices,
                                      inject_layer, n_rounds, total_steps,
                                      seed, grad_accum, tag, results_dir,
                                      benchmark_name, lambda_sparse=0.001,
                                      gate_hidden=128, initial_alpha=0.05,
                                      dora_rank=8):
    """Train hybrid controller on a standard benchmark loader (HF dict format)."""
    random.seed(seed)
    torch.manual_seed(seed)

    choice_ids = get_choice_tokens(tokenizer, n_choices)
    choice_ids_t = torch.tensor(choice_ids, device=device)

    controller = HybridDoRAController(
        frozen_llm=base_model, sites=sites, n_choices=n_choices,
        inject_layer=inject_layer,
        gate_hidden=gate_hidden, initial_alpha=initial_alpha,
        rank=64, d_state=512, n_slots=8,
        tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    # Param accounting
    ctrl_only = sum(
        p.numel() for n, p in controller.named_parameters()
        if p.requires_grad and not n.startswith('gate_head')
        and not (n.startswith('sites.') or 'frozen_llm' in n)
    )
    # Enumerate trainable DoRA params (not attached to controller directly —
    # they're inside the wrapped model).
    dora_only = count_dora_params(sites)
    gate_head_only = sum(p.numel() for p in controller.gate_head.parameters()
                         if p.requires_grad)
    total_trainable = sum(p.numel() for p in controller.parameters() if p.requires_grad)
    # Note: DoRA params are inside the frozen_llm submodule, so they ARE
    # counted in controller.parameters() because frozen_llm is a child of
    # the controller. We subtract and re-add explicitly for clarity.
    print(f'  Hybrid controller:', flush=True)
    print(f'    ctrl_non_gate  = {(total_trainable - gate_head_only - dora_only)/1e6:.2f}M', flush=True)
    print(f'    DoRA (r={dora_rank}, {len(sites)} sites) = {dora_only/1e6:.2f}M', flush=True)
    print(f'    gate_head      = {gate_head_only/1e6:.2f}M', flush=True)
    print(f'    TOTAL trainable= {total_trainable/1e6:.2f}M', flush=True)

    trainable_params = [p for p in controller.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.05)
    warmup = 200

    def lr_sched(s):
        if s < warmup:
            return s / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    # Benchmark-dict vs SpatialEval-list data handling
    spatial_mode = isinstance(data, list)
    if spatial_mode:
        task_list, train_idx, eval_idx = data  # (list, train_idx, eval_idx)
    else:
        train_data = data['train']
        test_data = data['test'][:500]
        random.shuffle(train_data)

    t0 = time.time()
    losses = []
    alpha_means = []
    optimizer.zero_grad(set_to_none=True)

    # Baseline (pure frozen LM — gates are cleared, so DoRA is bypassed)
    print(f'  Computing baseline...', flush=True)
    clear_gate_alphas(sites)
    controller.eval()
    baseline_correct = 0
    with torch.no_grad():
        if spatial_mode:
            eval_items = [(task_list[i], i) for i in eval_idx]
        else:
            eval_items = [(s, idx) for idx, s in enumerate(test_data)]
        for sample, _idx in eval_items:
            text = sample['text'][:1500] + "\nAnswer:"
            enc = tokenizer(text, return_tensors='pt', truncation=True,
                            max_length=1900).to(device)
            out = base_model(enc['input_ids'])
            logits = out.logits[:, -1, choice_ids_t]
            pred = logits.argmax(dim=-1).item()
            if spatial_mode:
                oracle = sample['oracle_option'].strip().upper()
                label = CHOICE_MAP.get(oracle[0], 0)
            else:
                label = sample['label']
            if pred == label:
                baseline_correct += 1
    n_eval = len(eval_items)
    baseline_acc = baseline_correct / max(n_eval, 1)
    print(f'  Baseline: {baseline_acc:.4f} ({baseline_correct}/{n_eval})', flush=True)

    # Train
    controller.train()
    print(f'  Training {total_steps} steps (GA={grad_accum})...', flush=True)
    for step in range(total_steps):
        if spatial_mode:
            sample = task_list[train_idx[step % len(train_idx)]]
            oracle = sample['oracle_option'].strip().upper()
            answer_label = CHOICE_MAP.get(oracle[0], 0)
            text = sample['text'][:1500]
        else:
            sample = train_data[step % len(train_data)]
            text = sample['text'][:1500]
            answer_label = sample['label']

        prompt_enc = tokenizer(text, return_tensors='pt', truncation=True,
                               max_length=1900).to(device)
        answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                               add_special_tokens=False).to(device)

        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
            answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])

        label_t = torch.tensor([answer_label], device=device, dtype=torch.long)
        all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_t,
                                    rounds=n_rounds)
        loss, lp = controller.compute_loss(all_cl, all_v, label_t,
                                            lambda_sparse=lambda_sparse)
        loss = loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(lp['final_ce'])
        if 'alpha_mean' in lp:
            alpha_means.append(lp['alpha_mean'])
        if (step + 1) % 1000 == 0:
            avg = sum(losses[-1000:]) / 1000
            am = sum(alpha_means[-1000:]) / max(1, len(alpha_means[-1000:]))
            print(f'  step {step+1}/{total_steps} | ce={avg:.4f} | '
                  f'alpha={am:.3f} | {time.time()-t0:.0f}s', flush=True)

    # Eval with K-scaling
    controller.eval()
    results = {'baseline': {'accuracy': baseline_acc, 'correct': baseline_correct,
                             'total': n_eval}}

    for er in [3, 5, 8]:
        correct = 0
        with torch.no_grad():
            for sample, _idx in eval_items:
                text = sample['text'][:1500]
                prompt_enc = tokenizer(text, return_tensors='pt', truncation=True,
                                       max_length=1900).to(device)
                answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                                       add_special_tokens=False).to(device)
                prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
                answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
                all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_t,
                                        rounds=er)
                pred = all_cl[-1].argmax(dim=-1).item()
                if spatial_mode:
                    oracle = sample['oracle_option'].strip().upper()
                    label = CHOICE_MAP.get(oracle[0], 0)
                else:
                    label = sample['label']
                if pred == label:
                    correct += 1
        acc = correct / max(n_eval, 1)
        delta = acc - baseline_acc
        results[f'rounds={er}'] = {'accuracy': acc, 'correct': correct,
                                    'total': n_eval, 'delta': delta}
        print(f'  rounds={er}: {acc:.4f} (delta={delta:+.4f})', flush=True)
    # Ensure we leave the model in the clean state for any later code.
    clear_gate_alphas(sites)

    os.makedirs(results_dir, exist_ok=True)
    result_data = {
        'tag': tag, 'benchmark': benchmark_name, 'n_choices': n_choices,
        'inject_layer': inject_layer, 'n_rounds': n_rounds,
        'total_steps': total_steps, 'seed': seed, 'grad_accum': grad_accum,
        'method': 'hybrid_dora',
        'dora_rank': dora_rank, 'n_sites': len(sites),
        'lambda_sparse': lambda_sparse,
        'ctrl_trainable_M': (total_trainable - gate_head_only - dora_only) / 1e6,
        'dora_trainable_M': dora_only / 1e6,
        'gate_head_trainable_M': gate_head_only / 1e6,
        'total_trainable_M': total_trainable / 1e6,
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(results_dir, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {tag}.json ({time.time()-t0:.0f}s)', flush=True)

    # Reset DoRA params for next benchmark (same wrapped model reused).
    with torch.no_grad():
        for _li, _name, g in sites:
            nn.init.normal_(g.A, std=0.01)
            g.B.zero_()
            g.mag.data.copy_(g.base_row_norms)
    del controller, optimizer, scheduler
    torch.cuda.empty_cache()
    return result_data


def parse_dora_layers(spec, n_layers):
    """Parse e.g. '20-31' or '0,4,8' into a tuple of layer indices."""
    spec = spec.strip()
    if '-' in spec and ',' not in spec:
        a, b = spec.split('-')
        return tuple(range(int(a), int(b) + 1))
    return tuple(int(x) for x in spec.split(','))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='models/full/Llama-3.1-8B-Instruct')
    parser.add_argument('--benchmarks', type=str, required=True,
                        help='Comma-separated benchmark names '
                             '(hellaswag,winogrande,boolq,spatialgrid,...)')
    parser.add_argument('--inject_layer', type=int, default=12)
    parser.add_argument('--n_rounds', type=int, default=5)
    parser.add_argument('--total_steps', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--dora_rank', type=int, default=8)
    parser.add_argument('--dora_layers', type=str, default='20-31',
                        help='Layer range (e.g. "20-31") or list (e.g. "0,4,8")')
    parser.add_argument('--dora_modules', type=str,
                        default='q_proj,o_proj,gate_proj,up_proj,down_proj')
    parser.add_argument('--lambda_sparse', type=float, default=0.001)
    parser.add_argument('--gate_hidden', type=int, default=128)
    parser.add_argument('--initial_alpha', type=float, default=0.05)
    parser.add_argument('--results_dir', type=str,
                        default='results/data/hybrid')
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
    n_layers = len(lm_model.layers)
    print(f'Model loaded ({n_layers} layers).', flush=True)

    # Wrap model with GatedDoRA at requested layers.
    dora_layers = parse_dora_layers(args.dora_layers, n_layers)
    dora_modules = tuple(m.strip() for m in args.dora_modules.split(','))
    print(f'Wrapping DoRA: layers={dora_layers} modules={dora_modules} rank={args.dora_rank}',
          flush=True)
    base_model, sites = wrap_model_with_gated_dora(
        base_model, target_modules=dora_modules,
        target_layers=dora_layers, rank=args.dora_rank,
    )
    print(f'  Wrapped {len(sites)} sites '
          f'({count_dora_params(sites)/1e6:.2f}M DoRA params).', flush=True)

    benchmarks = [b.strip() for b in args.benchmarks.split(',') if b.strip()]
    model_short = 'inst' if 'instruct' in args.model.lower() else 'base'

    overall_t0 = time.time()
    for i, bench_name in enumerate(benchmarks):
        print(f'\n{"=" * 70}', flush=True)
        print(f'[{i + 1}/{len(benchmarks)}] {bench_name}', flush=True)
        print(f'{"=" * 70}', flush=True)

        try:
            if bench_name in BENCHMARK_LOADERS:
                data, n_choices = BENCHMARK_LOADERS[bench_name]()
            elif bench_name in ('spatialgrid', 'mazenav', 'spatialmap', 'spatialreal'):
                task_list, train_idx, eval_idx, n_choices = load_spatialeval_task(bench_name)
                print(f'  {bench_name}: {len(train_idx)} train, {len(eval_idx)} eval',
                      flush=True)
                data = [task_list, train_idx, eval_idx]  # flag for spatial_mode
            else:
                print(f'  Unknown benchmark: {bench_name}', flush=True)
                continue

            tag = (f'hybrid_{model_short}_{bench_name}_L{args.inject_layer}_'
                   f'r{args.dora_rank}_{args.total_steps // 1000}k_s{args.seed}')

            train_and_eval_hybrid_benchmark(
                base_model, tokenizer, lm_model, sites,
                data=data, n_choices=n_choices,
                inject_layer=args.inject_layer,
                n_rounds=args.n_rounds,
                total_steps=args.total_steps,
                seed=args.seed,
                grad_accum=args.grad_accum,
                tag=tag,
                results_dir=args.results_dir,
                benchmark_name=bench_name,
                lambda_sparse=args.lambda_sparse,
                gate_hidden=args.gate_hidden,
                initial_alpha=args.initial_alpha,
                dora_rank=args.dora_rank,
            )
        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()

    print(f'\n=== All benchmarks done in {time.time() - overall_t0:.0f}s ===',
          flush=True)


if __name__ == '__main__':
    main()
