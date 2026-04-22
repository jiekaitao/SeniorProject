"""
ARR-PSRT Training: 3-phase curriculum with mixed data.

Phase A (25%): K=2, uniform routing, only expert FFNs + re-reader + scratchpad train
Phase B (42%): K={2,3}, soft routing, all params
Phase C (33%): K={1-4}, top-2 routing, halting enabled

Data: 45% FineWeb-Edu + 20% OpenMathInstruct + 20% NaturalReasoning/ARC + 15% general reasoning

Usage:
    python train_arr.py --size 172m --total_steps 12000
"""

import os, sys, json, time, math, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from arr_psrt import ARRPSRT, ARRConfig, create_arr_psrt


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok


def mixed_stream(tokenizer, seq_len, batch_size):
    from datasets import load_dataset
    general = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                                split='train', streaming=True))
    math_ds = iter(load_dataset('nvidia/OpenMathInstruct-2', split='train', streaming=True))
    science = iter(load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='train', streaming=True))

    token_buffer = []

    def get_text(source):
        try:
            if source == 'general':
                return next(general).get('text', '')
            elif source == 'math':
                ex = next(math_ds)
                return f"Problem: {ex.get('problem', '')}\nSolution: {ex.get('generated_solution', '')}"
            elif source == 'science':
                ex = next(science)
                q = ex.get('question', '')
                choices = ex.get('choices', {})
                labels = choices.get('label', [])
                texts = choices.get('text', [])
                ak = ex.get('answerKey', '')
                cs = ' '.join(f"({l}) {t}" for l, t in zip(labels, texts))
                at = next((t for l, t in zip(labels, texts) if l == ak), '')
                return f"Question: {q}\n{cs}\nAnswer: ({ak}) {at}"
        except (StopIteration, Exception):
            return ''

    while True:
        r = random.random()
        text = get_text('general' if r < 0.45 else ('math' if r < 0.65 else 'science'))
        if not text or len(text) < 30:
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
            yield t[:, :-1], t[:, 1:]


_eval_cache = None  # cache eval tokens across calls

def evaluate_ppl(model, tokenizer, device, n=20, seq_len=512, bs=4):
    global _eval_cache
    model.eval()

    # Cache eval data to avoid repeated network calls (streaming can hang)
    if _eval_cache is None or len(_eval_cache) < (seq_len + 1) * bs * n * 2 + 1000:
        import signal
        def _timeout_handler(signum, frame):
            raise TimeoutError("Dataset load timed out")
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(120)  # 2 min timeout
            from datasets import load_dataset
            ds = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                                   split='train', streaming=True))
            buf = []
            target = (seq_len + 1) * bs * n * 2 + 1000
            while len(buf) < target:
                ex = next(ds)
                t = ex.get('text', '')
                if t and len(t) > 30:
                    toks = tokenizer.encode(t, add_special_tokens=False,
                                            truncation=True, max_length=seq_len * 2)
                    toks.append(tokenizer.eos_token_id)
                    buf.extend(toks)
            signal.alarm(0)
            _eval_cache = buf
            print("  [eval: cached data from fineweb]", flush=True)
        except Exception as e:
            signal.alarm(0)
            print(f"  [eval: dataset failed ({e}), using random tokens]", flush=True)
            vocab_size = tokenizer.vocab_size or 32000
            _eval_cache = torch.randint(10, vocab_size, ((seq_len + 1) * bs * n * 2 + 1000,)).tolist()

    buf = list(_eval_cache)  # copy so we don't consume the cache
    results = {}
    for K in [1, 2]:
        total_loss = total_tok = 0
        with torch.no_grad():
            for i in range(n):
                batch = []
                for _ in range(bs):
                    batch.append(buf[:seq_len + 1])
                    buf = buf[seq_len:]
                t = torch.tensor(batch, dtype=torch.long).to(device)
                _, loss, _ = model(t[:, :-1], labels=t[:, :-1], fixed_K=K)
                total_loss += loss.item() * t[:, 1:].numel()
                total_tok += t[:, 1:].numel()
        results[f'K={K}'] = math.exp(total_loss / max(total_tok, 1))
    model.train()
    return results


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = create_arr_psrt(args.size)
    model = model.to(device)
    tokenizer = get_tokenizer()

    total = args.total_steps
    phase_a_end = int(total * 0.25)
    phase_b_end = int(total * 0.67)
    warmup = int(total * 0.05)

    # Separate param groups: expert/reread/scratch vs shared
    expert_names = {'expert_ffn', 'reread', 'scratch', 'compressor'}
    expert_params = []
    shared_params = []
    for name, p in model.named_parameters():
        if any(en in name for en in expert_names):
            expert_params.append(p)
        else:
            shared_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': expert_params, 'lr': args.lr},
        {'params': shared_params, 'lr': args.lr},
    ], weight_decay=0.01, betas=(0.9, 0.95))

    def lr_schedule(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_schedule, lr_schedule])

    save_dir = args.save_dir or 'psrt/checkpoints/arr_psrt'
    os.makedirs(save_dir, exist_ok=True)

    # Resume from checkpoint if provided
    start_step = 0
    if args.resume:
        print(f'  Loading checkpoint: {args.resume}', flush=True)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt['model_state'], strict=False)
        if missing:
            print(f'  New params (initialized fresh): {missing}', flush=True)
            print(f'  Skipping optimizer state (architecture changed)', flush=True)
        elif 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        start_step = ckpt.get('step', 0)
        # Advance scheduler to match resumed step
        for _ in range(start_step):
            scheduler.step()
        print(f'  Resumed from step {start_step} (phase={ckpt.get("phase", "?")})', flush=True)

    print(f'\n=== ARR-PSRT Training ===', flush=True)
    print(f'  {model.count_params()/1e6:.0f}M params', flush=True)
    print(f'  Steps: {total} (A:{phase_a_end} B:{phase_b_end} C:{total})', flush=True)
    print(f'  Prompt bank: {cfg.prompt_bank_size}, Scratchpad: {cfg.scratchpad_size}', flush=True)
    print(f'  Data: 45% general + 20% math + 35% science/reasoning', flush=True)

    gen = mixed_stream(tokenizer, args.seq_len, args.batch_size)
    model.train()

    # Set requires_grad based on current phase
    if args.joint:
        # Joint training: all params active from step 0, shared at scaled LR
        for p in model.parameters():
            p.requires_grad = True
        if args.shared_scale is not None:
            optimizer.param_groups[1]['lr'] = args.lr * args.shared_scale
        print(f'  Joint training: all params active, shared LR={optimizer.param_groups[1]["lr"]:.2e}', flush=True)
    elif start_step < phase_a_end:
        # Phase A: freeze shared params
        for p in shared_params:
            p.requires_grad = False
        for p in expert_params:
            p.requires_grad = True
    else:
        # Phase B/C: all params trainable
        for p in model.parameters():
            p.requires_grad = True
        shared_scale = 0.01 if sum(p.numel() for p in shared_params) > 500_000_000 else 0.3
        optimizer.param_groups[1]['lr'] = args.lr * shared_scale

    step = start_step
    running_loss = 0
    running_ek = 0
    nan_count = 0
    t0 = time.time()
    best_ppl = float('inf')
    shared_target_lr = 0.0
    shared_warmup_steps = 500

    for input_ids, labels in gen:
        if step >= total:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        if step == phase_a_end:
            # Save checkpoint before Phase B transition
            torch.save({
                'step': step, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': cfg.__dict__, 'phase': 'A_end',
            }, f'{save_dir}/phase_a_end.pt')
            print(f'  --- SAVED phase_a_end checkpoint ---', flush=True)
            print(f'\n=== Phase B: K={{2,3}} + soft routing ===', flush=True)
            # Zero-init router weights: prevents one-hot collapse from random init
            # (Phase A used uniform routing so router was never trained — weights are random)
            for n, p in model.named_parameters():
                if 'router' in n:
                    nn.init.zeros_(p)
                    state = optimizer.state.get(p)
                    if state:
                        if 'exp_avg' in state:
                            state['exp_avg'].zero_()
                        if 'exp_avg_sq' in state:
                            state['exp_avg_sq'].zero_()
            print(f'  Router weights: ZEROED (uniform start)', flush=True)
            if args.freeze_shared:
                print(f'  Shared params: FROZEN (--freeze_shared)', flush=True)
                shared_target_lr = 0.0
                shared_warmup_steps = 1
            else:
                for p in shared_params:
                    p.requires_grad = True
                # Reset optimizer state for shared params to clear stale momentum
                for p in shared_params:
                    state = optimizer.state.get(p)
                    if state:
                        if 'exp_avg' in state:
                            state['exp_avg'].zero_()
                        if 'exp_avg_sq' in state:
                            state['exp_avg_sq'].zero_()
                if args.shared_scale is not None:
                    shared_scale = args.shared_scale
                else:
                    shared_scale = 0.0001 if sum(p.numel() for p in shared_params) > 500_000_000 else 0.3
                # Start at 0 LR — will ramp up over 2000 steps
                optimizer.param_groups[1]['lr'] = 0.0
                shared_target_lr = args.lr * shared_scale
                shared_warmup_steps = 2000
                print(f'  Shared params target LR: {shared_target_lr:.2e} (warmup over {shared_warmup_steps} steps)', flush=True)

        if step == phase_b_end:
            print(f'\n=== Phase C: K={{1-4}} + top-2 + halting ===', flush=True)

        if args.uniform_only:
            # Pure re-reading: K=2, uniform routing, no phase transitions
            fixed_K = 2
            train_halting = False
            uniform_routing = True
            phase = 'A'
        elif step < phase_a_end:
            fixed_K = 2
            train_halting = False
            uniform_routing = True
            phase = 'A'
        elif step < phase_b_end:
            # Phase B1 (first 2000 steps): K=2 only — stabilize routing before adding depth
            # Phase B2 (rest): gradually ramp K=3 probability from 0 to 0.4
            b_progress = step - phase_a_end
            if b_progress < 2000:
                fixed_K = 2  # B1: no K=3 yet
            else:
                k3_ramp = min((b_progress - 2000) / 2000, 1.0)  # 0→1 over next 2000 steps
                k3_prob = 0.40 * k3_ramp
                fixed_K = random.choices([2, 3], weights=[1.0 - k3_prob, k3_prob])[0]
            train_halting = False
            uniform_routing = False
            phase = 'B'
        else:
            fixed_K = random.choices([1, 2, 3, 4], weights=[0.15, 0.40, 0.30, 0.15])[0]
            train_halting = True
            uniform_routing = False
            phase = 'C'

        # Gradual shared LR warmup during Phase B transition
        if phase_a_end <= step < phase_a_end + shared_warmup_steps:
            warmup_frac = (step - phase_a_end) / shared_warmup_steps
            optimizer.param_groups[1]['lr'] = shared_target_lr * warmup_frac

        # Phase B: soft routing with high temperature to prevent one-hot collapse
        # Phase C: hard top-k routing (router should be stable by then)
        soft_routing = (phase == 'B')
        # Temperature starts at 3.0 and anneals to 1.0 over Phase B
        if phase == 'B' and step >= phase_a_end:
            pb_frac = (step - phase_a_end) / max(phase_b_end - phase_a_end, 1)
            router_temp = 3.0 - 2.0 * pb_frac  # 3.0 → 1.0
        else:
            router_temp = 1.0
        logits, loss, aux = model(input_ids, labels=labels, fixed_K=fixed_K,
                                   train_halting=train_halting, uniform_routing=uniform_routing,
                                   soft_routing=soft_routing, router_temp=router_temp)

        if train_halting:
            loss = loss + 0.0005 * aux['expected_steps']

        loss.backward()
        # Skip step if loss is NaN (gradient recovery)
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            nan_count += 1
            if nan_count >= 50:
                print(f'  !!! ABORT: {nan_count} consecutive NaN at step {step} !!!', flush=True)
                break
            if nan_count % 10 == 0:
                print(f'  [NaN skip #{nan_count} at step {step}]', flush=True)
            continue
        nan_count = 0  # reset on successful step
        # Tighter clipping for Phase B to prevent NaN from unfrozen params
        clip_val = 0.1 if step >= phase_a_end else 1.0
        # Gradient norm monitoring per component (before clipping, every 500 steps in Phase B)
        if phase == 'B' and step % 500 == 0:
            gnorms = {}
            for gname, gparams in [('expert', expert_params), ('shared', shared_params)]:
                total_norm = 0.0
                for p in gparams:
                    if p.grad is not None:
                        total_norm += p.grad.data.float().norm().item() ** 2
                gnorms[gname] = total_norm ** 0.5
            for tag in ['router', 'reread_attn', 'scratch']:
                tn = 0.0
                for n, p in model.named_parameters():
                    if tag in n and p.grad is not None:
                        tn += p.grad.data.float().norm().item() ** 2
                gnorms[tag] = tn ** 0.5
            gn_str = ' | '.join(f'{k}={v:.4f}' for k, v in gnorms.items())
            print(f'  [GRAD] step {step}: {gn_str}', flush=True)
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val, error_if_nonfinite=True)
        except RuntimeError:
            # Gradient already NaN — skip this step
            optimizer.zero_grad(set_to_none=True)
            nan_count += 1
            if nan_count >= 50:
                print(f'  !!! ABORT: {nan_count} consecutive NaN grads at step {step} !!!', flush=True)
                break
            if nan_count % 10 == 0:
                print(f'  [NaN grad #{nan_count} at step {step}]', flush=True)
            continue
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        es = aux['expected_steps']
        running_ek += es.item() if isinstance(es, torch.Tensor) else es
        step += 1

        if step % 100 == 0:
            avg_loss = running_loss / 100
            avg_ek = running_ek / 100
            elapsed = time.time() - t0
            route = aux.get('route_dist', [])
            route_str = ' '.join(f'{r:.2f}' for r in route) if route else '?'
            print(f'  step {step:6d} P{phase} | loss={avg_loss:.4f} | E[K]={avg_ek:.2f} | '
                  f'K={fixed_K} | alpha={aux["alpha"]:.3f} | route=[{route_str}] | '
                  f'{elapsed:.0f}s', flush=True)
            running_loss = running_ek = 0

        if step % 5000 == 0 and step > 0:
            torch.save({
                'step': step, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': cfg.__dict__, 'phase': phase,
            }, f'{save_dir}/step_{step}.pt')
            print(f'  --- SAVED checkpoint step {step} ---', flush=True)

        if step % 2000 == 0:
            ppl = evaluate_ppl(model, tokenizer, device, n=15,
                               seq_len=args.seq_len, bs=args.batch_size)
            delta = ppl['K=2'] - ppl['K=1']
            print(f'  --- EVAL step {step}: PPL K=1={ppl["K=1"]:.2f} K=2={ppl["K=2"]:.2f} '
                  f'(delta={delta:+.2f}) ---', flush=True)
            if ppl['K=1'] < best_ppl:
                best_ppl = ppl['K=1']
                torch.save({
                    'step': step, 'model_state': model.state_dict(),
                    'config': cfg.__dict__, 'ppl': ppl,
                }, f'{save_dir}/best.pt')
                print(f'  --- SAVED best ---', flush=True)

    # Final
    torch.save({'step': step, 'model_state': model.state_dict(),
                'config': cfg.__dict__}, f'{save_dir}/final.pt')
    ppl = evaluate_ppl(model, tokenizer, device, n=30, seq_len=args.seq_len, bs=args.batch_size)
    delta = ppl['K=2'] - ppl['K=1']
    print(f'\n=== Complete: PPL K=1={ppl["K=1"]:.2f} K=2={ppl["K=2"]:.2f} (delta={delta:+.2f}) ===',
          flush=True)
    print(f'  Best PPL: {best_ppl:.2f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='172m', choices=['172m', '1b'])
    parser.add_argument('--total_steps', type=int, default=12000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--freeze_shared', action='store_true', help='Keep shared params frozen for entire run')
    parser.add_argument('--uniform_only', action='store_true', help='Use uniform routing for entire run (no learned routing)')
    parser.add_argument('--shared_scale', type=float, default=None, help='Override shared LR scale (default: auto based on model size)')
    parser.add_argument('--joint', action='store_true', help='Joint training: all params from step 0, no Phase A freeze')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
