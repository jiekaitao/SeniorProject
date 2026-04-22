"""
Attention-Only vs FFN-Only Control — GPT-5.4's #1 recommendation.

Tests the FFN re-retrieval hypothesis in deliberation:
- attention_only: thought tokens affect attention but are zeroed before FFN
- ffn_only: thought tokens affect FFN but are zeroed before attention
- full: normal deliberation (control)

If attention-only improves cross-task generalization and FFN-only harms,
we have strong evidence for FFN interference in deliberation.

Also tests grad accumulation (effective batch 16) to reduce seed variance.
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation, RMSNorm

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/attn_ffn_control'
CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


class SelectiveDeliberation(RecurrentDeliberation):
    """
    Deliberation controller that selectively applies thought tokens
    to attention-only or FFN-only pathways via hooks.
    """
    def __init__(self, frozen_llm, mode='full', **kwargs):
        super().__init__(frozen_llm, **kwargs)
        self.mode = mode  # 'full', 'attention_only', 'ffn_only'

    def forward_frozen_round(self, prompt_emb, thought_emb, answer_emb):
        """Run frozen decoder with selective thought injection."""
        lm_model = self.frozen_llm.model

        # Assemble input
        dec_input = torch.cat([prompt_emb, thought_emb, answer_emb], dim=1)
        T = dec_input.shape[1]
        pos_ids = torch.arange(T, device=dec_input.device).unsqueeze(0)
        pos_emb = lm_model.rotary_emb(dec_input, pos_ids)

        p_len = prompt_emb.shape[1]
        t_len = thought_emb.shape[1]
        think_slice = slice(p_len, p_len + t_len)

        h = dec_input
        tapped_pools = []

        # Store original thought positions for selective masking
        hooks = []

        if self.mode == 'attention_only':
            # Zero thought positions AFTER attention, BEFORE FFN
            def make_post_attn_hook(layer_module):
                def hook_fn(module, input, output):
                    # output[0] is hidden states after attention
                    out = output[0] if isinstance(output, tuple) else output
                    # Zero the think slots so FFN doesn't see them
                    out_mod = out.clone()
                    out_mod[:, think_slice] = h_before_layer[:, think_slice]
                    if isinstance(output, tuple):
                        return (out_mod,) + output[1:]
                    return out_mod
                return hook_fn

        elif self.mode == 'ffn_only':
            # Zero thought positions BEFORE attention, restore BEFORE FFN
            pass  # More complex, handle below

        for i, layer in enumerate(lm_model.layers):
            if self.mode == 'attention_only':
                # Save state before this layer
                h_before_layer = h.clone()

                # Run attention part of the layer
                # We need to intercept between attention and FFN
                # Most transformer layers: residual_pre -> attn -> residual_post -> ffn
                # For Llama: layer(h) does attn + ffn internally

                # Simpler approach: run full layer, then mask think-slot FFN contribution
                # by restoring think-slot hidden states to post-attention values

                # Register hook on the MLP/FFN to zero think slot gradients
                if hasattr(layer, 'mlp'):
                    def make_ffn_hook(saved_h):
                        def hook_fn(module, input, output):
                            # FFN output for think slots should be zero
                            # input[0] is the input to FFN (post-attn residual)
                            # We want to zero the FFN's effect on think slots
                            mask = torch.ones_like(output)
                            mask[:, think_slice] = 0.0
                            return output * mask
                        return hook_fn
                    hk = layer.mlp.register_forward_hook(make_ffn_hook(h))
                    hooks.append(hk)

            elif self.mode == 'ffn_only':
                # Zero think slots before attention, restore before FFN
                if hasattr(layer, 'self_attn'):
                    def make_attn_hook():
                        def hook_fn(module, input, output):
                            # Zero attention's effect on think slots
                            out = output[0] if isinstance(output, tuple) else output
                            mask = torch.ones_like(out)
                            mask[:, think_slice] = 0.0
                            result = out * mask
                            if isinstance(output, tuple):
                                return (result,) + output[1:]
                            return result
                        return hook_fn
                    hk = layer.self_attn.register_forward_hook(make_attn_hook())
                    hooks.append(hk)

            h = layer(h, position_embeddings=pos_emb)

            if i in self.tapped_layers:
                tapped_pools.append(h.mean(dim=1))

        # Remove hooks
        for hk in hooks:
            hk.remove()

        h = lm_model.norm(h)
        logits = self.frozen_llm.lm_head(h)
        think_h = h[:, think_slice]

        return logits, think_h, tapped_pools


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


def run_experiment(mode, task, seed, total_steps, n_rounds, grad_accum,
                   tokenizer, base_model, data, train_idx, eval_idx, choice_ids):
    tag = f'attn_ffn_{mode}_{task}_r{n_rounds}_ga{grad_accum}_seed{seed}'
    random.seed(seed)
    torch.manual_seed(seed)

    print(f'\n{"="*60}', flush=True)
    print(f'  Mode: {mode} | Task: {task} | Rounds: {n_rounds} | GA: {grad_accum} | Seed: {seed}', flush=True)
    print(f'{"="*60}', flush=True)

    controller = SelectiveDeliberation(
        frozen_llm=base_model, mode=mode,
        d_state=512, n_slots=8, tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    print(f'  Params: {controller.count_trainable():,}', flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05
    )
    warmup = 200
    eff_steps = total_steps // grad_accum
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
        total_loss = total_loss / grad_accum

        if total_loss.requires_grad:
            total_loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in controller.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        losses_hist.append(loss_parts['final_ce'])

        if (step + 1) % 200 == 0:
            avg = sum(losses_hist[-200:]) / len(losses_hist[-200:])
            rN = (all_cl[-1].argmax(dim=-1) == label_tensor).float().item()
            print(f'  step {step+1} | ce={avg:.4f} | correct={rN:.0f} | {time.time()-t0:.0f}s', flush=True)

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
        'tag': tag, 'method': 'selective_deliberation',
        'mode': mode, 'task': task,
        'n_rounds': n_rounds, 'grad_accum': grad_accum,
        'seed': seed, 'total_steps': total_steps,
        'results': results,
        'final_loss': sum(losses_hist[-50:]) / max(len(losses_hist[-50:]), 1),
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
    parser.add_argument('--modes', type=str, default='full,attention_only,ffn_only')
    parser.add_argument('--task', type=str, default='mazenav')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seeds', type=str, default='42,7')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--grad_accum', type=int, default=1)
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
                          args.grad_accum, tokenizer, base_model, data,
                          train_idx, eval_idx, choice_ids)

    print('\n=== All selective control experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
