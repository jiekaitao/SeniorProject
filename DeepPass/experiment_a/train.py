"""
Experiment A: Band-Recurrent Llama 3.1 8B

GPT-5.4 Pro recipe:
- Frozen Llama 3.1 8B base
- Replay attention on layers 12-15 with LoRA rank 16
- Mixed K training: 50% K=1, 50% K=2
- Hard-token routing: replay only top 25% entropy tokens
- Loss: CE(K=2) + 0.5*CE(K=1) + 0.5*KL_easy + MLP_gate_penalty
"""

import os, sys, time, math, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from band_recurrent import BandRecurrentLlama


def get_data_stream(tokenizer, seq_len, batch_size):
    """Reasoning-rich data mix: math + science + general."""
    general = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                                split='train', streaming=True))
    math_ds = iter(load_dataset('nvidia/OpenMathInstruct-2', split='train', streaming=True))

    token_buffer = []

    def get_text():
        r = random.random()
        try:
            if r < 0.5:
                ex = next(math_ds)
                return f"Problem: {ex.get('problem', '')}\nSolution: {ex.get('generated_solution', '')}"
            else:
                return next(general).get('text', '')
        except (StopIteration, Exception):
            return ''

    while True:
        text = get_text()
        if not text or len(text) < 50:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                  max_length=seq_len * 2)
        tokens.append(tokenizer.eos_token_id)
        token_buffer.extend(tokens)

        while len(token_buffer) >= (seq_len + 1) * batch_size:
            batch = []
            for _ in range(batch_size):
                chunk = token_buffer[:seq_len + 1]
                token_buffer = token_buffer[seq_len:]
                batch.append(chunk)
            t = torch.tensor(batch, dtype=torch.long)
            yield t


def evaluate(model, tokenizer, device, n_batches=20, seq_len=512, bs=2):
    """Evaluate PPL at K=1 and K=2, plus hard-token NLL."""
    model.eval()
    ds = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                           split='train', streaming=True))
    buf = []
    target = (seq_len + 1) * bs * n_batches * 2 + 500
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
        total_loss = total_tok = 0
        buf_copy = list(buf)
        with torch.no_grad():
            for _ in range(n_batches):
                batch = []
                for _ in range(bs):
                    batch.append(buf_copy[:seq_len + 1])
                    buf_copy = buf_copy[seq_len:]
                t = torch.tensor(batch, dtype=torch.long).to(device)
                _, loss = model(t[:, :-1], labels=t[:, :-1], K=K)
                total_loss += loss.item() * t[:, 1:].numel()
                total_tok += t[:, 1:].numel()
        results[f'K={K}'] = math.exp(total_loss / max(total_tok, 1))

    model.train()
    return results


def train(replay_layers=None):
    device = torch.device('cuda')
    model_path = 'models/full/Llama-3.1-8B'

    print(f'Loading {model_path}...', flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map='auto',
    )
    print(f'  Loaded in {time.time()-t0:.0f}s', flush=True)

    # Wrap with band-recurrent replay
    layers = replay_layers or (12, 13, 14, 15)
    model = BandRecurrentLlama(
        base_model,
        replay_layer_ids=layers,
        lora_rank=16,
        max_extra_passes=1,
    )

    n_trainable = model.count_trainable()
    n_total = model.count_total()
    print(f'  Total params: {n_total/1e9:.2f}B', flush=True)
    print(f'  Trainable: {n_trainable:,} ({n_trainable/n_total*100:.3f}%)', flush=True)

    # Optimizer: only trainable params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-2, weight_decay=0.01, betas=(0.9, 0.95),
    )

    total_steps = 10000
    warmup = 500
    eval_every = 500

    def lr_schedule(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    save_dir = 'experiment_a/checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n=== Experiment A: Band-Recurrent Llama 3.1 8B ===', flush=True)
    print(f'  Replay layers: 12-15 (4 of 32)', flush=True)
    print(f'  LoRA rank: 16', flush=True)
    print(f'  Hard-token fraction: 25%', flush=True)
    print(f'  Steps: {total_steps}', flush=True)
    print(f'  Data: 50% math + 50% general', flush=True)

    data_stream = get_data_stream(tokenizer, seq_len=1024, batch_size=2)
    model.train()

    step = 0
    running_loss = 0
    t0 = time.time()
    best_ppl = float('inf')

    for batch in data_stream:
        if step >= total_steps:
            break
        batch = batch.to(device)
        input_ids = batch[:, :-1]
        labels = batch[:, :-1]

        # Always K=2 — gates start at zero so it's effectively K=1 until they learn
        K = 2
        _, loss = model(input_ids, labels=labels, K=K, hard_frac=0.25)

        # MLP gate penalty (skip if MVP without separate mlp_gate)
        mlp_pen = 0.0
        for extra in model.extra_layers.values():
            if hasattr(extra, 'mlp_gate'):
                mlp_pen = mlp_pen + extra.mlp_gate ** 2
        if mlp_pen != 0.0:
            loss = loss + 1e-4 * mlp_pen

        # Only backward if loss has gradient (K=1 on frozen model has no grad path)
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        step += 1

        if step % 50 == 0:
            avg = running_loss / 50
            elapsed = time.time() - t0
            gates = [f'{torch.tanh(e.gate).item():.4f}' for e in model.extra_layers.values()]
            mix = torch.tanh(model.mix_gate.to(torch.float32)).item()
            print(f'  step {step:5d} | loss={avg:.4f} | K={K} | mix={mix:.4f} | '
                  f'gates=[{",".join(gates)}] | {elapsed:.0f}s',
                  flush=True)
            running_loss = 0

        if step % eval_every == 0:
            ppl = evaluate(model, tokenizer, device, n_batches=15, seq_len=1024, bs=2)
            delta = ppl['K=2'] - ppl['K=1']
            print(f'  --- EVAL step {step}: PPL K=1={ppl["K=1"]:.2f} K=2={ppl["K=2"]:.2f} '
                  f'(delta={delta:+.2f}) ---', flush=True)
            if ppl['K=1'] < best_ppl:
                best_ppl = ppl['K=1']
                torch.save({
                    'step': step,
                    'extra_layers': {k: v.state_dict() for k, v in model.extra_layers.items()},
                    'inj_gate': model.inj_gate.data,
                    'mix_gate': model.mix_gate.data,
                    'ppl': ppl,
                }, f'{save_dir}/best.pt')
                print(f'  --- SAVED best ---', flush=True)

    # Final eval
    ppl = evaluate(model, tokenizer, device, n_batches=30, seq_len=1024, bs=2)
    delta = ppl['K=2'] - ppl['K=1']
    print(f'\n=== Complete: PPL K=1={ppl["K=1"]:.2f} K=2={ppl["K=2"]:.2f} (delta={delta:+.2f}) ===',
          flush=True)
    print(f'  Best PPL: {best_ppl:.2f}', flush=True)


if __name__ == '__main__':
    import sys
    layers = None
    if len(sys.argv) > 1:
        layers = tuple(int(x) for x in sys.argv[1].split(','))
    train(replay_layers=layers)
