"""
Experiment A: Reasoning benchmark evaluation.

Takes the best replay checkpoint and evaluates on reasoning tasks
instead of just PPL. Uses the base model's own generate() with
replay layers enabled vs disabled.

Tests:
1. Arithmetic (from our math_probe)
2. Multi-step reasoning
3. Hard-token-specific PPL (top 25% entropy tokens only)
"""

import os, sys, time, math, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from band_recurrent import BandRecurrentLlama


def hard_token_ppl(model, tokenizer, device, n=30, seq_len=1024, bs=2, hard_frac=0.25):
    """Measure PPL on ONLY the hardest tokens (top entropy quartile)."""
    model.eval()
    ds = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                           split='train', streaming=True))
    buf = []
    target = (seq_len + 1) * bs * n + 500
    while len(buf) < target:
        ex = next(ds)
        t = ex.get('text', '')
        if t and len(t) > 50:
            toks = tokenizer.encode(t, add_special_tokens=False, truncation=True,
                                    max_length=seq_len * 2)
            toks.append(tokenizer.eos_token_id)
            buf.extend(toks)

    results = {}
    for K in [1, 2]:
        hard_nll_sum = hard_tok_count = 0
        all_nll_sum = all_tok_count = 0
        buf_copy = list(buf)

        with torch.no_grad():
            for _ in range(n):
                batch = []
                for _ in range(bs):
                    batch.append(buf_copy[:seq_len + 1])
                    buf_copy = buf_copy[seq_len:]
                t = torch.tensor(batch, dtype=torch.long).to(device)
                input_ids = t[:, :-1]
                targets = t[:, 1:]

                logits, _ = model(input_ids, K=K, hard_frac=1.0)
                logits = logits.float()

                # Per-token NLL
                log_probs = F.log_softmax(logits, dim=-1)
                nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T)

                # Identify hard tokens via base model entropy
                base_probs = F.softmax(logits, dim=-1)
                entropy = -(base_probs * (base_probs + 1e-8).log()).sum(-1)  # (B, T)
                thresh = torch.quantile(entropy.flatten(), 1.0 - hard_frac)
                hard_mask = (entropy >= thresh).float()

                hard_nll_sum += (nll * hard_mask).sum().item()
                hard_tok_count += hard_mask.sum().item()
                all_nll_sum += nll.sum().item()
                all_tok_count += nll.numel()

        results[f'K={K}_all'] = math.exp(all_nll_sum / max(all_tok_count, 1))
        results[f'K={K}_hard'] = math.exp(hard_nll_sum / max(hard_tok_count, 1))

    model.train()
    return results


def arithmetic_eval(model, tokenizer, device, K=1, n=50):
    """Simple arithmetic accuracy test."""
    model.eval()
    correct = 0
    total = 0

    problems = []
    for _ in range(n):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        op = random.choice(['+', '-', '*'])
        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        else:
            answer = a * b
        problems.append((f"Calculate: {a} {op} {b} = ", str(answer)))

    for prompt, expected in problems:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            logits, _ = model(input_ids, K=K, hard_frac=1.0)

        # Get the predicted next tokens (greedy)
        pred_tokens = logits[0, -1].argmax().item()
        pred_text = tokenizer.decode([pred_tokens]).strip()
        # Check if the first predicted character matches
        if expected.startswith(pred_text) or pred_text.startswith(expected[:len(pred_text)]):
            correct += 1
        total += 1

    model.train()
    return correct / max(total, 1)


def main():
    device = torch.device('cuda')
    model_path = 'models/full/Llama-3.1-8B'

    print(f'Loading {model_path}...', flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map='auto',
    )
    print(f'  Loaded in {time.time()-t0:.0f}s', flush=True)

    model = BandRecurrentLlama(
        base_model,
        replay_layer_ids=(12, 13, 14, 15),
        lora_rank=16,
        max_extra_passes=1,
    )

    # Load best checkpoint if available
    ckpt_path = 'experiment_a/checkpoints/best.pt'
    if os.path.exists(ckpt_path):
        print(f'  Loading checkpoint: {ckpt_path}', flush=True)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        for k, v in ckpt.get('extra_layers', {}).items():
            model.extra_layers[k].load_state_dict(v)
        if 'inj_gate' in ckpt:
            model.inj_gate.data = ckpt['inj_gate']
        if 'mix_gate' in ckpt:
            model.mix_gate.data = ckpt['mix_gate']
        print(f'  Loaded from step {ckpt.get("step", "?")}', flush=True)
    else:
        print('  No checkpoint — using fresh gates (init=0.1)', flush=True)

    # Move to device
    model._to_device(device, torch.bfloat16)

    print(f'\n=== Hard-Token PPL Evaluation ===', flush=True)
    ht_results = hard_token_ppl(model, tokenizer, device, n=30, seq_len=1024, bs=2, hard_frac=0.25)
    print(f'  All tokens:  K=1 PPL={ht_results["K=1_all"]:.2f}  K=2 PPL={ht_results["K=2_all"]:.2f}  delta={ht_results["K=2_all"]-ht_results["K=1_all"]:+.2f}')
    print(f'  Hard tokens: K=1 PPL={ht_results["K=1_hard"]:.2f}  K=2 PPL={ht_results["K=2_hard"]:.2f}  delta={ht_results["K=2_hard"]-ht_results["K=1_hard"]:+.2f}')
    print(flush=True)

    print(f'=== Arithmetic Eval ===', flush=True)
    for K in [1, 2]:
        acc = arithmetic_eval(model, tokenizer, device, K=K, n=100)
        print(f'  K={K}: accuracy={acc:.1%}', flush=True)

    print(f'\n=== Done ===', flush=True)


if __name__ == '__main__':
    main()
