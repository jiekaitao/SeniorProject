"""
LLM-to-TRM: Convert any pre-trained LLM into a Thinking Recursive Model

Takes a pre-trained model and surgically adds the PSRT memory/reasoning split:
1. Insert proj_m and proj_r projection layers BEFORE the core
2. Insert combine layer AFTER the core
3. Freeze ALL base weights, train only the 3 new projection layers
4. During K=2+, memory (m) is frozen, only reasoning (r) iterates

The key insight: we DON'T need to train from scratch. The existing LLM layers
already know how to process hidden states. We just need to teach the projections
to split the state into "what to remember" vs "what to think about."

Architecture:
    [Base layers 0..core_start)              ← frozen prelude
    proj_m(h) → m_0                          ← NEW, trainable
    proj_r(h) → r_0                          ← NEW, trainable
    [Core layers] × K: r = (1-α)r + α(Core(r+m₀)-m₀)  ← frozen core, iterated
    combine(m_0, r) → h                      ← NEW, trainable
    [Base layers core_end..N)                ← frozen coda

Only ~8M new params for a 4096-dim model. Everything else frozen.

Usage:
    python llm_to_trm.py --model <path> --name <name> --core_start 10 --core_end 13
"""

import os, sys, json, time, math, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe, _load_questions
from transformers import AutoModelForCausalLM, AutoTokenizer


class LayerIdxWrapper(nn.Module):
    def __init__(self, layer, new_idx):
        super().__init__()
        self.layer = layer
        self.new_layer_idx = new_idx
        self.orig_idx = layer.layer_idx
        self.orig_attn = layer.self_attn.layer_idx if hasattr(layer, 'self_attn') else None

    def forward(self, *args, **kwargs):
        self.layer.layer_idx = self.new_layer_idx
        if hasattr(self.layer, 'self_attn'):
            self.layer.self_attn.layer_idx = self.new_layer_idx
        try:
            return self.layer(*args, **kwargs)
        finally:
            self.layer.layer_idx = self.orig_idx
            if self.orig_attn is not None:
                self.layer.self_attn.layer_idx = self.orig_attn

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)


def get_inner(model):
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    return inner


class TRMProjections(nn.Module):
    """The surgical addition: memory/reasoning projections + combine."""

    def __init__(self, d_model, alpha_init=0.5):
        super().__init__()
        # Project hidden state into memory and reasoning channels
        self.proj_m = nn.Linear(d_model, d_model, bias=False)
        self.proj_r = nn.Linear(d_model, d_model, bias=False)

        # Combine memory + reasoning back into hidden state
        self.combine = nn.Linear(2 * d_model, d_model, bias=False)

        # Learned mixing parameter
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid → 0.5

        # Init projections near identity (start close to base model behavior)
        nn.init.eye_(self.proj_m.weight)
        nn.init.eye_(self.proj_r.weight)
        # Init combine to roughly average m and r
        with torch.no_grad():
            self.combine.weight.zero_()
            self.combine.weight[:, :d_model] = 0.5 * torch.eye(d_model)
            self.combine.weight[:, d_model:] = 0.5 * torch.eye(d_model)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)


def forward_trm(model, input_ids, projections, core_start, core_end, K=2,
                use_cache=False, is_gemma=False):
    """
    Forward pass with TRM memory/reasoning split.

    Manually runs through layers with the split-state recursion:
    1. Prelude: layers [0, core_start) — standard forward
    2. Split: m = proj_m(h), r = proj_r(h)
    3. Core × K: r = (1-α)r + α(Core(r + m) - m)
    4. Combine: h = combine([m, r])
    5. Coda: layers [core_end, N) — standard forward
    """
    inner = get_inner(model)
    layers = list(inner.layers)

    # Get embeddings
    if hasattr(inner, 'embed_tokens'):
        h = inner.embed_tokens(input_ids)
    elif hasattr(inner, 'wte'):
        h = inner.wte(input_ids)
    else:
        h = inner.embed(input_ids)

    B, L = input_ids.shape
    dtype = h.dtype
    position_ids = torch.arange(L, device=h.device).unsqueeze(0).expand(B, -1)

    # Precompute rotary embeddings (LLaMA/Mistral need this)
    position_embeddings = None
    if hasattr(inner, 'rotary_emb'):
        position_embeddings = inner.rotary_emb(h, position_ids)

    def call_layer(layer, hidden):
        """Call a decoder layer with the right kwargs."""
        kwargs = {'position_ids': position_ids}
        if position_embeddings is not None:
            kwargs['position_embeddings'] = position_embeddings
        out = layer(hidden, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    # === Prelude ===
    for i in range(core_start):
        h = call_layer(layers[i], h)

    # === Split into memory and reasoning ===
    m_0 = projections.proj_m(h)
    r = projections.proj_r(h)
    alpha = projections.alpha

    # === Recursive Core ===
    for k in range(K):
        h_core = r + m_0
        for i in range(core_start, core_end):
            h_core = call_layer(layers[i], h_core)

        # Extract reasoning update
        r_new = h_core - m_0
        r = (1.0 - alpha) * r + alpha * r_new

    # === Combine ===
    h = projections.combine(torch.cat([m_0, r], dim=-1))

    # === Coda ===
    for i in range(core_end, len(layers)):
        h = call_layer(layers[i], h)

    # === Final norm + LM head ===
    if hasattr(inner, 'norm'):
        h = inner.norm(h)
    elif hasattr(inner, 'ln_f'):
        h = inner.ln_f(h)

    if hasattr(model, 'lm_head'):
        logits = model.lm_head(h)
    else:
        logits = model.output(h)

    return logits


def evaluate(model, tokenizer, device, tag="", projections=None, core_start=None,
             core_end=None, K=1, is_gemma=False):
    """Evaluate with optional TRM forward."""
    model.eval()
    if projections:
        projections.eval()
    eq_all = _load_questions()

    def gen(p):
        inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=512).to(device)
        kw = {k: v for k, v in inputs.items() if k != 'token_type_ids'}
        with torch.no_grad():
            if projections and K > 1:
                # Can't use generate() with custom forward — use greedy manual decode
                input_ids = kw['input_ids']
                for _ in range(64):
                    logits = forward_trm(model, input_ids, projections,
                                         core_start, core_end, K, is_gemma=is_gemma)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                return tokenizer.decode(input_ids[0][kw['input_ids'].shape[1]:],
                                        skip_special_tokens=True)
            else:
                out = model.generate(**kw, max_new_tokens=64, do_sample=False, use_cache=False)
                return tokenizer.decode(out[0][kw['input_ids'].shape[1]:],
                                        skip_special_tokens=True)

    def gen_long(p):
        inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=512).to(device)
        kw = {k: v for k, v in inputs.items() if k != 'token_type_ids'}
        with torch.no_grad():
            if projections and K > 1:
                input_ids = kw['input_ids']
                for _ in range(128):
                    logits = forward_trm(model, input_ids, projections,
                                         core_start, core_end, K, is_gemma=is_gemma)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                return tokenizer.decode(input_ids[0][kw['input_ids'].shape[1]:],
                                        skip_special_tokens=True)
            else:
                out = model.generate(**kw, max_new_tokens=128, do_sample=False, use_cache=False)
                return tokenizer.decode(out[0][kw['input_ids'].shape[1]:],
                                        skip_special_tokens=True)

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    print(f'  {tag}: math={math_r["score"]:.4f} eq={eq_r["score"]:.1f} '
          f'combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
    return {'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}


def train_trm(model, tokenizer, device, projections, core_start, core_end,
              max_steps=500, lr=1e-4, is_gemma=False):
    """Train only the TRM projection layers."""
    from datasets import load_dataset

    params = list(projections.parameters())
    n_params = sum(p.numel() for p in params)
    print(f'  Trainable TRM params: {n_params:,}', flush=True)

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps,
                                                            eta_min=lr * 0.1)

    ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
    model.train()
    projections.train()

    step = 0
    running_loss = 0
    running_loss_k1 = 0
    t0 = time.time()
    token_buffer = []

    for example in ds:
        if step >= max_steps:
            break
        text = example.get('text', '')
        if not text or len(text) < 100:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=1024)
        if tokenizer.eos_token_id:
            tokens.append(tokenizer.eos_token_id)
        token_buffer.extend(tokens)

        while len(token_buffer) >= 513 and step < max_steps:
            chunk = torch.tensor([token_buffer[:513]], dtype=torch.long).to(device)
            token_buffer = token_buffer[512:]
            input_ids = chunk[:, :-1]
            labels = chunk[:, 1:]

            # Curriculum: K=1 (50%), K=2 (35%), K=3 (15%)
            K = random.choices([1, 2, 3], weights=[0.50, 0.35, 0.15])[0]

            logits = forward_trm(model, input_ids, projections,
                                  core_start, core_end, K, is_gemma=is_gemma)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            if K == 1:
                running_loss_k1 += loss.item()

            step += 1
            if step % 50 == 0:
                avg = running_loss / 50
                alpha = projections.alpha.item()
                elapsed = time.time() - t0
                print(f'  step {step:4d} | loss={avg:.4f} | K={K} | '
                      f'alpha={alpha:.3f} | lr={scheduler.get_last_lr()[0]:.2e} | '
                      f'{elapsed:.0f}s', flush=True)
                running_loss = 0

    print(f'  Done: {step} steps, alpha={projections.alpha.item():.3f}, '
          f'{time.time()-t0:.0f}s', flush=True)
    model.eval()
    projections.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--cache_dir', default='/blue/cis4914/jietao/hf_cache')
    parser.add_argument('--core_start', type=int, default=10)
    parser.add_argument('--core_end', type=int, default=13)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    SAVE_DIR = f'sirt/recursion_ft/{args.name}_trm'
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda')

    is_gemma = 'gemma' in args.model.lower()

    # Load model
    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, device_map='auto',
        dtype=torch.bfloat16, trust_remote_code=True,
    )
    model_dtype = next(model.parameters()).dtype

    # Freeze ALL base weights
    for param in model.parameters():
        param.requires_grad = False

    # Create TRM projections
    d_model = getattr(model.config, 'hidden_size', 4096)
    projections = TRMProjections(d_model).to(device=device, dtype=model_dtype)
    print(f'TRM projections: {sum(p.numel() for p in projections.parameters()):,} params', flush=True)
    print(f'  alpha init: {projections.alpha.item():.3f}', flush=True)

    # === Baseline K=1 (standard model, no projections) ===
    print('\n=== Baseline K=1 ===', flush=True)
    baseline = evaluate(model, tokenizer, device, 'baseline K=1')

    # === Pre-ft K=1 through projections (should be ~baseline due to identity init) ===
    print('\n=== Pre-ft K=1 through TRM projections ===', flush=True)
    pre_k1 = evaluate(model, tokenizer, device, 'pre-ft TRM K=1',
                       projections=projections, core_start=args.core_start,
                       core_end=args.core_end, K=1, is_gemma=is_gemma)

    # === Pre-ft K=2 through projections (untrained, near-identity) ===
    print('\n=== Pre-ft K=2 through TRM projections ===', flush=True)
    pre_k2 = evaluate(model, tokenizer, device, 'pre-ft TRM K=2',
                       projections=projections, core_start=args.core_start,
                       core_end=args.core_end, K=2, is_gemma=is_gemma)

    # === Train TRM projections ===
    print(f'\n=== Training TRM Projections ({args.max_steps} steps) ===', flush=True)
    train_trm(model, tokenizer, device, projections, args.core_start, args.core_end,
              max_steps=args.max_steps, lr=args.lr, is_gemma=is_gemma)

    # === Post-ft K=1 ===
    print('\n=== Post-ft K=1 ===', flush=True)
    post_k1 = evaluate(model, tokenizer, device, 'post-ft TRM K=1',
                        projections=projections, core_start=args.core_start,
                        core_end=args.core_end, K=1, is_gemma=is_gemma)

    # === Post-ft K=2 ===
    print('\n=== Post-ft K=2 ===', flush=True)
    post_k2 = evaluate(model, tokenizer, device, 'post-ft TRM K=2',
                        projections=projections, core_start=args.core_start,
                        core_end=args.core_end, K=2, is_gemma=is_gemma)

    # === Post-ft K=3 ===
    print('\n=== Post-ft K=3 ===', flush=True)
    post_k3 = evaluate(model, tokenizer, device, 'post-ft TRM K=3',
                        projections=projections, core_start=args.core_start,
                        core_end=args.core_end, K=3, is_gemma=is_gemma)

    # === Summary ===
    print(f'\n{"=" * 60}', flush=True)
    print(f'LLM-to-TRM SUMMARY -- {args.name}', flush=True)
    print(f'{"=" * 60}', flush=True)
    print(f'  Baseline K=1 (orig):   {baseline["combined"]:.2f}', flush=True)
    print(f'  Pre-ft TRM K=1:        {pre_k1["combined"]:.2f} ({pre_k1["combined"]-baseline["combined"]:+.2f})', flush=True)
    print(f'  Pre-ft TRM K=2:        {pre_k2["combined"]:.2f} ({pre_k2["combined"]-baseline["combined"]:+.2f})', flush=True)
    print(f'  Post-ft TRM K=1:       {post_k1["combined"]:.2f} ({post_k1["combined"]-baseline["combined"]:+.2f})', flush=True)
    print(f'  Post-ft TRM K=2:       {post_k2["combined"]:.2f} ({post_k2["combined"]-baseline["combined"]:+.2f})', flush=True)
    print(f'  Post-ft TRM K=3:       {post_k3["combined"]:.2f} ({post_k3["combined"]-baseline["combined"]:+.2f})', flush=True)
    print(f'  Learned alpha:         {projections.alpha.item():.3f}', flush=True)
    print(f'  K=2 recursion benefit: {post_k2["combined"]-post_k1["combined"]:+.2f}', flush=True)
    print('COMPLETE', flush=True)

    # Save
    ckpt = {
        'projections_state': projections.state_dict(),
        'config': vars(args),
        'results': {
            'baseline': baseline, 'pre_k1': pre_k1, 'pre_k2': pre_k2,
            'post_k1': post_k1, 'post_k2': post_k2, 'post_k3': post_k3,
        },
    }
    torch.save(ckpt, f'{SAVE_DIR}/checkpoint.pt')
    with open(f'{SAVE_DIR}/results.json', 'w') as f:
        json.dump(ckpt['results'], f, indent=2)
    print(f'Saved to {SAVE_DIR}/', flush=True)


if __name__ == '__main__':
    main()
