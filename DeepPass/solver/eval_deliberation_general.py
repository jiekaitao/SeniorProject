"""
Generalized Recurrent Deliberation Controller — Multi-Model + Multi-Task.

Tests generalization across:
  - Models: Llama 3.1 8B, Llama 3.1 8B Instruct, Gemma 3 27B-IT
  - Tasks: SpatialEval mazenav, spatialmap, spatialgrid, spatialreal

Uses hooks-based tapping (works for ALL model architectures, including
Gemma 3's sliding window attention which breaks manual layer loops).
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/spatialeval'
CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


# ============================================================
# Model-agnostic helpers
# ============================================================
def get_model_internals(model):
    """Auto-detect model architecture and return (layers, embed_tokens, norm, lm_head, d_model)."""
    config = model.config

    # Try different architectures
    if hasattr(model, 'model'):
        inner = model.model
        # Gemma 3/4: model.model.language_model
        if hasattr(inner, 'language_model'):
            lm = inner.language_model
            return lm.layers, lm.embed_tokens, lm.norm, model.lm_head, lm.config.hidden_size
        # Llama: model.model
        elif hasattr(inner, 'layers'):
            return inner.layers, inner.embed_tokens, inner.norm, model.lm_head, config.hidden_size

    raise ValueError(f"Unknown model architecture: {type(model)}")


def get_choice_token_ids(tokenizer):
    ids = []
    for c in ['A', 'B', 'C', 'D']:
        toks = tokenizer.encode(f" {c}", add_special_tokens=False)
        ids.append(toks[0])
    return ids


# ============================================================
# Hooks-based Recurrent Deliberation Controller
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


class HooksDeliberation(nn.Module):
    """
    Hooks-based deliberation controller that works with ANY transformer architecture.
    Uses forward hooks instead of manual layer loops.
    """
    def __init__(self, base_model, d_model, vocab_size, n_layers,
                 d_state=512, n_slots=8, tapped_layers=None, topk_vocab=64):
        super().__init__()
        self.base_model = base_model
        self.d_model_llm = d_model
        self.n_slots = n_slots
        self.topk_vocab = topk_vocab

        if tapped_layers is None:
            # Default: evenly spaced across the model
            step = max(1, n_layers // 4)
            tapped_layers = [step, 2*step, 3*step]
        self.tapped_layers = sorted(tapped_layers)
        self.n_tap = len(self.tapped_layers)

        # Get model internals
        self.layers, self.embed_tokens, self.norm, self.lm_head, _ = get_model_internals(base_model)

        # Freeze base model
        for p in base_model.parameters():
            p.requires_grad = False

        # Controller state
        self.z0 = nn.Parameter(torch.randn(1, n_slots, d_state) * 0.02)

        # Read: tapped hidden states + slot hidden states + choice logits + uncertainty
        read_dim = self.n_tap * d_model + n_slots * d_model + 4 + 2
        self.read_proj = nn.Sequential(
            nn.Linear(read_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, n_slots * d_state),
        )

        # State update
        self.state_norm = RMSNorm(d_state)
        self.state_gate = nn.Parameter(torch.tensor(0.1))

        # Write: z -> sparse vocab superposition
        self.to_vocab_logits = nn.Linear(d_state, vocab_size, bias=False)
        nn.init.normal_(self.to_vocab_logits.weight, std=0.01)

        # Verifier
        self.verifier = nn.Sequential(
            nn.Linear(self.n_tap * d_model + n_slots * d_model + 4, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )

    def latent_to_thought_embs(self, z):
        E = self.embed_tokens.weight  # (V, d_model)
        logits = self.to_vocab_logits(z)
        vals, idx = logits.topk(self.topk_vocab, dim=-1)
        probs = F.softmax(vals, dim=-1)
        chosen_embs = E[idx]
        return (probs.unsqueeze(-1) * chosen_embs).sum(dim=-2)

    def forward_with_hooks(self, inputs_embeds, think_start, think_end):
        """Run frozen model with hooks to capture hidden states at tapped layers."""
        tapped_outputs = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is a tuple, first element is hidden states
                h = output[0] if isinstance(output, tuple) else output
                tapped_outputs[layer_idx] = h.mean(dim=1).to(h.dtype)
            return hook_fn

        # Register hooks
        for i in self.tapped_layers:
            if i < len(self.layers):
                h = self.layers[i].register_forward_hook(make_hook(i))
                hooks.append(h)

        # Forward pass through full model
        with torch.no_grad():
            # Need to handle position IDs for the full model
            # Use inputs_embeds directly
            outputs = self.base_model(inputs_embeds=inputs_embeds, use_cache=False,
                                       output_hidden_states=True)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Get logits and hidden states at think slot positions
        logits = outputs.logits
        # Get final hidden state at think positions
        last_hidden = outputs.hidden_states[-1]  # (B, T, d_model)
        think_h = last_hidden[:, think_start:think_end]  # (B, n_slots, d_model)

        tapped_pools = [tapped_outputs[i] for i in self.tapped_layers if i in tapped_outputs]

        return logits, think_h, tapped_pools

    def build_features(self, think_h, tapped_pools, choice_logits):
        dtype = think_h.dtype
        probs = choice_logits.float().softmax(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        top2 = probs.topk(2, dim=-1).values
        margin = top2[:, :1] - top2[:, 1:2]
        feat = torch.cat(
            [think_h.flatten(1)] + tapped_pools +
            [probs.to(dtype), entropy.to(dtype), margin.to(dtype)],
            dim=-1
        )
        return feat

    def forward(self, prompt_emb, answer_emb, choice_ids, rounds=2):
        B = prompt_emb.shape[0]
        z = self.z0.expand(B, -1, -1).clone()
        all_choice_logits = []
        all_verify = []

        p_len = prompt_emb.shape[1]

        for r in range(rounds):
            thought_emb = self.latent_to_thought_embs(z).to(prompt_emb.dtype)
            t_len = thought_emb.shape[1]

            # Assemble: [prompt | THINK slots | answer_prefix]
            dec_input = torch.cat([prompt_emb, thought_emb, answer_emb], dim=1)

            # Forward with hooks
            logits, think_h, tapped_pools = self.forward_with_hooks(
                dec_input, p_len, p_len + t_len
            )

            # Answer logits (last position before generation)
            ans_logits = logits[:, -1, choice_ids]
            all_choice_logits.append(ans_logits)

            # Features
            feat = self.build_features(think_h, tapped_pools, ans_logits)

            # Verifier
            verify_probs = ans_logits.float().softmax(dim=-1).to(think_h.dtype)
            verify_feat = torch.cat(
                [think_h.flatten(1)] + tapped_pools + [verify_probs], dim=-1
            )
            verify = self.verifier(verify_feat)
            all_verify.append(verify)

            # Update state
            if r < rounds - 1:
                delta = self.read_proj(feat).view(B, self.n_slots, -1)
                z = self.state_norm(z + self.state_gate * delta)

        return all_choice_logits, all_verify

    def compute_loss(self, all_choice_logits, all_verify, answer_labels,
                     lambda_v=0.5, lambda_p=0.1, delta_p=0.1):
        final_ce = F.cross_entropy(all_choice_logits[-1].float(), answer_labels)
        verify_loss = torch.tensor(0.0, device=answer_labels.device, dtype=torch.float32)
        for r in range(len(all_choice_logits)):
            pred_correct = (all_choice_logits[r].argmax(dim=-1) == answer_labels).float()
            verify_loss = verify_loss + F.binary_cross_entropy_with_logits(
                all_verify[r].float().squeeze(-1), pred_correct)
        verify_loss = verify_loss / len(all_choice_logits)
        progress_loss = torch.tensor(0.0, device=answer_labels.device, dtype=torch.float32)
        if len(all_choice_logits) > 1:
            first_ce = F.cross_entropy(all_choice_logits[0].float(), answer_labels)
            progress_loss = F.relu(final_ce - first_ce + delta_p)
        return final_ce + lambda_v * verify_loss + lambda_p * progress_loss, {
            'final_ce': final_ce.item(), 'verify_loss': verify_loss.item(),
            'progress_loss': progress_loss.item(),
        }

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Data loading
# ============================================================
def load_spatialeval(task='mazenav'):
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    if task == 'all':
        data = list(ds)
    else:
        data = [s for s in ds if s['id'].startswith(task)]
    print(f'Loaded {len(data)} {task} samples', flush=True)
    return data


# ============================================================
# Training + Eval
# ============================================================
def run_experiment(model_path, task, n_rounds, n_slots, seed, total_steps,
                   tokenizer, base_model, data, train_idx, eval_idx, choice_ids,
                   custom_tapped=None):
    model_name = os.path.basename(model_path)
    tag = f'general_{model_name}_{task}_r{n_rounds}_s{n_slots}_seed{seed}'
    random.seed(seed)
    torch.manual_seed(seed)

    layers, embed_tokens, norm, lm_head, d_model = get_model_internals(base_model)
    n_layers = len(layers)
    vocab_size = embed_tokens.weight.shape[0]

    # Select tapped layers
    if custom_tapped:
        tapped = [t for t in custom_tapped if t < n_layers]
    else:
        step = max(1, n_layers // 4)
        tapped = [step, 2*step, min(3*step, n_layers-1)]

    print(f'\n{"="*60}', flush=True)
    print(f'  Model: {model_name} ({d_model}d, {n_layers}L, {vocab_size}V)', flush=True)
    print(f'  Task: {task} | rounds={n_rounds} | slots={n_slots} | seed={seed}', flush=True)
    print(f'  Tapped layers: {tapped}', flush=True)
    print(f'{"="*60}', flush=True)

    controller = HooksDeliberation(
        base_model, d_model=d_model, vocab_size=vocab_size, n_layers=n_layers,
        d_state=512, n_slots=n_slots, tapped_layers=tapped, topk_vocab=64
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
            prompt_emb = embed_tokens(prompt_enc['input_ids'])
            answer_emb = embed_tokens(answer_enc['input_ids'])

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

        if (step + 1) % 100 == 0:
            avg = sum(losses_hist[-100:]) / len(losses_hist[-100:])
            print(f'  step {step+1} | ce={avg:.4f} | {time.time()-t0:.0f}s', flush=True)

    # ========== EVAL ==========
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
                    prompt_emb = embed_tokens(prompt_enc['input_ids'])
                    answer_emb = embed_tokens(answer_enc['input_ids'])
                    all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=eval_rounds)
                    pred = all_cl[-1].argmax(dim=-1).item()
                if pred == answer_label_val:
                    correct += 1

        acc = correct / n_eval
        results[f'rounds={eval_rounds}'] = {'accuracy': acc, 'correct': correct, 'total': n_eval}
        label = 'baseline' if eval_rounds == 0 else f'rounds={eval_rounds}'
        print(f'  {label}: {acc:.4f} ({correct}/{n_eval})', flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': tag, 'method': 'general_deliberation',
        'model': model_name, 'model_path': model_path,
        'd_model': d_model, 'n_layers': n_layers,
        'task': task, 'n_rounds': n_rounds, 'n_slots': n_slots,
        'tapped_layers': tapped,
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
    parser.add_argument('--model', type=str, default='models/full/Llama-3.1-8B')
    parser.add_argument('--task', type=str, default='mazenav')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--slots', type=int, default=8)
    parser.add_argument('--seeds', type=str, default='42,7')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--tapped', type=str, default='auto',
                        help='Comma-separated tapped layer indices, or "auto"')
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(',')]

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    print(f'Model loaded.', flush=True)

    choice_ids = get_choice_token_ids(tokenizer)
    print(f'Choice tokens: {dict(zip(["A","B","C","D"], choice_ids))}', flush=True)

    data = load_spatialeval(args.task)
    random.seed(0)
    indices = list(range(len(data)))
    random.shuffle(indices)
    split = min(1000, len(data) * 2 // 3)
    train_idx = indices[:split]
    eval_idx = indices[split:]
    print(f'Split: {len(train_idx)} train, {len(eval_idx)} eval', flush=True)

    custom_tapped = None
    if args.tapped != 'auto':
        custom_tapped = [int(x) for x in args.tapped.split(',')]
        print(f'Custom tapped layers: {custom_tapped}', flush=True)

    for seed in seeds:
        run_experiment(args.model, args.task, args.rounds, args.slots, seed,
                      args.steps, tokenizer, base_model, data, train_idx, eval_idx, choice_ids,
                      custom_tapped=custom_tapped)

    print('\n=== All experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
