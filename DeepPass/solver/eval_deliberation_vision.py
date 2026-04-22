"""
Vision Deliberation Controller — Extending deliberation to vision-language models.

Tests whether our recurrent deliberation controller can improve spatial reasoning
on VISUAL benchmarks (VSR, GQA) using frozen vision-language models (LLaVA, PaLiGemma2).

Architecture:
  1. Frozen VLM processes image + text → fused embeddings
  2. Controller reads hidden states, writes thought tokens
  3. Thoughts injected at mid-layer of the LM backbone
  4. Iterative refinement across rounds

Supported models:
  - LLaVA-NeXT (Mistral-7B backbone): hidden=4096, layers=32
  - PaLiGemma2-10B (Gemma2 backbone): hidden=3584, layers=42
  - Llama 3.2 11B Vision: hidden=4096, layers=32 (when available)
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation, RMSNorm

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/vision'


def detect_vlm_architecture(model):
    """Detect the VLM architecture and return access paths.

    Structure for LLaVA/PaLiGemma ForConditionalGeneration:
      model.model.vision_tower       — vision encoder
      model.model.multi_modal_projector — projects vision → LM space
      model.model.language_model     — bare LM (MistralModel/Gemma2Model)
        .embed_tokens, .layers, .norm
      model.lm_head                  — output projection
    """
    cfg = model.config
    model_type = getattr(cfg, 'model_type', '')

    # The composite model wrapper
    composite = model.model  # LlavaNextModel or PaliGemmaModel

    # Language model backbone (bare model, NOT ForCausalLM)
    lm_inner = composite.language_model  # MistralModel / Gemma2Model

    d_model = cfg.text_config.hidden_size
    n_layers = cfg.text_config.num_hidden_layers
    vocab_size = cfg.text_config.vocab_size

    if model_type == 'llava_next':
        arch_type = 'llava'
    elif model_type == 'paligemma':
        arch_type = 'paligemma'
    elif model_type == 'mllama':
        arch_type = 'mllama'
    else:
        raise ValueError(f'Unsupported VLM type: {model_type}')

    return {
        'type': arch_type,
        'composite': composite,
        'lm_inner': lm_inner,
        'layers': lm_inner.layers,
        'embed_tokens': lm_inner.embed_tokens,
        'norm': lm_inner.norm,
        'lm_head': model.lm_head,
        'rotary_emb': lm_inner.rotary_emb if hasattr(lm_inner, 'rotary_emb') else None,
        'd_model': d_model,
        'n_layers': n_layers,
        'vocab_size': vocab_size,
    }


def get_fused_embeddings(model, processor, arch, image, text):
    """Get the fused image+text embeddings from the VLM.
    Returns embeddings in the LM's hidden space (after vision projection).

    Uses model.model (the composite) for vision_tower and projector,
    and arch['embed_tokens'] for text embeddings.
    """
    composite = arch['composite']

    if arch['type'] == 'llava':
        # Use the model's own _merge method to handle variable-resolution patches
        inputs = processor(text=text, images=image, return_tensors='pt').to(device)
        with torch.no_grad():
            # Let the full model compute up to inputs_embeds via its internal logic
            outputs = model.model(
                input_ids=inputs.get('input_ids'),
                pixel_values=inputs.get('pixel_values'),
                image_sizes=inputs.get('image_sizes'),
                output_hidden_states=True,
                return_dict=True,
            )
            # Return the first hidden state (post-embedding, pre-layers)
            # This is the fused image+text embedding
            return outputs.hidden_states[0]

    elif arch['type'] == 'paligemma':
        inputs = processor(text=text, images=image, return_tensors='pt').to(device)
        with torch.no_grad():
            input_ids = inputs['input_ids']
            pixel_values = inputs.get('pixel_values', None)

            if pixel_values is not None:
                image_outputs = composite.vision_tower(
                    pixel_values.to(dtype=next(composite.vision_tower.parameters()).dtype)
                )
                img_features = image_outputs.last_hidden_state
                img_features = composite.multi_modal_projector(img_features)

                text_embs = arch['embed_tokens'](input_ids)
                n_img = img_features.shape[1]
                result = text_embs.clone()
                result[:, :n_img] = img_features[:, :n_img].to(text_embs.dtype)
                return result
            else:
                return arch['embed_tokens'](input_ids)

    elif arch['type'] == 'mllama':
        inputs = processor(text=text, images=image, return_tensors='pt').to(device)
        with torch.no_grad():
            return arch['embed_tokens'](inputs['input_ids'])

    else:
        raise ValueError(f'Unsupported: {arch["type"]}')


class VisionMidLayerDeliberation(nn.Module):
    """Deliberation controller adapted for vision-language models.
    Injects thought tokens at a mid-layer of the LM backbone."""

    def __init__(self, arch, inject_layer=12, d_state=512, n_slots=8,
                 tapped_layers=None, topk_vocab=64, rank=64):
        super().__init__()
        self.arch = arch
        self.inject_layer = inject_layer
        self.d_model = arch['d_model']
        self.n_slots = n_slots
        self.topk_vocab = topk_vocab

        n_layers = arch['n_layers']
        if tapped_layers is None:
            # Auto-select: ~1/4, 1/2, 3/4 of depth
            tapped_layers = (n_layers // 4, n_layers // 2, 3 * n_layers // 4)
        self.tapped_layers = set(tapped_layers)
        self.tapped_list = sorted(tapped_layers)
        n_tap = len(tapped_layers)

        # Initial controller state
        self.z0 = nn.Parameter(torch.randn(1, n_slots, d_state) * 0.02)

        # Read projection
        read_dim = n_tap * self.d_model + n_slots * self.d_model + 4 + 2
        self.read_proj = nn.Sequential(
            nn.Linear(read_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, n_slots * d_state),
        )

        # State update
        self.state_norm = RMSNorm(d_state)
        self.state_gate = nn.Parameter(torch.tensor(0.1))

        # Write: z → sparse vocab superposition + lowrank
        self.to_vocab_logits = nn.Linear(d_state, arch['vocab_size'], bias=False)
        nn.init.normal_(self.to_vocab_logits.weight, std=0.01)

        # Lowrank writer
        self.to_lowrank = nn.Linear(d_state, rank, bias=False)
        self.U = nn.Parameter(torch.randn(rank, self.d_model) * 0.02)
        nn.init.normal_(self.to_lowrank.weight, std=0.01)

        # Verifier
        self.verifier = nn.Sequential(
            nn.Linear(n_tap * self.d_model + n_slots * self.d_model + 4, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )

        # Number of choices (2 for binary, 4 for ABCD)
        self.n_choices = 4  # will be set dynamically

    def latent_to_thought_embs(self, z):
        E = self.arch['embed_tokens'].weight
        logits = self.to_vocab_logits(z)
        vals, idx = logits.topk(self.topk_vocab, dim=-1)
        probs = F.softmax(vals, dim=-1)
        chosen_embs = E[idx]
        vocab_part = (probs.unsqueeze(-1) * chosen_embs).sum(dim=-2)
        lowrank_part = self.to_lowrank(z) @ self.U
        return vocab_part + 0.12 * lowrank_part

    def forward_frozen_round(self, prompt_emb, thought_emb, answer_emb):
        layers = self.arch['layers']
        norm = self.arch['norm']
        lm_head = self.arch['lm_head']
        rotary_emb = self.arch['rotary_emb']

        # Run first N layers with just prompt + answer (no thoughts)
        dec_input = torch.cat([prompt_emb, answer_emb], dim=1)
        T = dec_input.shape[1]
        pos_ids = torch.arange(T, device=dec_input.device).unsqueeze(0)

        if rotary_emb is not None:
            pos_emb = rotary_emb(dec_input, pos_ids)
        else:
            pos_emb = None

        h = dec_input
        tapped_pools = []

        for i, layer in enumerate(layers):
            if i == self.inject_layer:
                # Inject thought tokens
                h = torch.cat([
                    h[:, :prompt_emb.shape[1]],
                    thought_emb.to(h.dtype),
                    h[:, prompt_emb.shape[1]:]
                ], dim=1)
                T_new = h.shape[1]
                pos_ids_new = torch.arange(T_new, device=h.device).unsqueeze(0)
                if rotary_emb is not None:
                    pos_emb = rotary_emb(h, pos_ids_new)

            if pos_emb is not None:
                out = layer(h, position_embeddings=pos_emb)
            else:
                out = layer(h)
            h = out[0] if isinstance(out, tuple) else out

            if i in self.tapped_layers:
                tapped_pools.append(h.mean(dim=1))

        h = norm(h)
        logits = lm_head(h)

        # Think slot hidden states
        if self.inject_layer < len(layers):
            p_len = prompt_emb.shape[1]
            t_len = thought_emb.shape[1]
            think_h = h[:, p_len:p_len + t_len]
        else:
            think_h = h[:, :self.n_slots]

        return logits, think_h, tapped_pools

    def forward(self, prompt_emb, answer_emb, choice_ids, rounds=2):
        B = prompt_emb.shape[0]
        z = self.z0.expand(B, -1, -1).clone()
        n_choices = choice_ids.shape[0]

        all_choice_logits = []
        all_verify = []

        for r in range(rounds):
            thought_emb = self.latent_to_thought_embs(z)
            logits, think_h, tapped_pools = self.forward_frozen_round(
                prompt_emb, thought_emb.to(prompt_emb.dtype), answer_emb
            )
            ans_logits = logits[:, -1, choice_ids]  # (B, n_choices)
            all_choice_logits.append(ans_logits)

            # Build features
            dtype = think_h.dtype
            probs = ans_logits.float().softmax(dim=-1)
            # Pad to 4 if fewer choices
            if n_choices < 4:
                pad = torch.zeros(B, 4 - n_choices, device=probs.device)
                probs_padded = torch.cat([probs, pad], dim=-1)
            else:
                probs_padded = probs[:, :4]

            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
            top2 = probs.topk(min(2, n_choices), dim=-1).values
            if top2.shape[-1] < 2:
                margin = top2[:, :1]
            else:
                margin = top2[:, :1] - top2[:, 1:2]

            feat = torch.cat(
                [think_h.flatten(1)] + tapped_pools +
                [probs_padded.to(dtype), entropy.to(dtype), margin.to(dtype)],
                dim=-1
            )

            # Verifier
            verify_feat = torch.cat(
                [think_h.flatten(1)] + tapped_pools + [probs_padded.to(dtype)],
                dim=-1
            )
            verify = self.verifier(verify_feat)
            all_verify.append(verify)

            if r < rounds - 1:
                delta = self.read_proj(feat).view(B, self.n_slots, -1)
                z = self.state_norm(z + self.state_gate * delta)

        return all_choice_logits, all_verify

    def compute_loss(self, all_choice_logits, all_verify, answer_labels,
                     lambda_v=0.5, lambda_p=0.1, delta_p=0.1):
        rounds = len(all_choice_logits)
        final_ce = F.cross_entropy(all_choice_logits[-1].float(), answer_labels)
        verify_loss = torch.tensor(0.0, device=answer_labels.device, dtype=torch.float32)
        for r in range(rounds):
            pred_correct = (all_choice_logits[r].argmax(dim=-1) == answer_labels).float()
            verify_loss = verify_loss + F.binary_cross_entropy_with_logits(
                all_verify[r].float().squeeze(-1), pred_correct
            )
        verify_loss = verify_loss / rounds
        progress_loss = torch.tensor(0.0, device=answer_labels.device, dtype=torch.float32)
        if rounds > 1:
            first_ce = F.cross_entropy(all_choice_logits[0].float(), answer_labels)
            progress_loss = F.relu(final_ce - first_ce + delta_p)
        total = final_ce + lambda_v * verify_loss + lambda_p * progress_loss
        return total, {
            'final_ce': final_ce.item(),
            'verify_loss': verify_loss.item(),
            'progress_loss': progress_loss.item(),
        }


def load_vsr_dataset():
    """Load VSR (Visual Spatial Reasoning) benchmark."""
    from datasets import load_dataset
    ds = load_dataset('cambridgeltl/vsr_random')
    print(f'  VSR loaded: train={len(ds["train"])}, val={len(ds["validation"])}, test={len(ds["test"])}')
    return ds


def load_image_safe(url_or_path, max_retries=2):
    """Load an image from URL or local path, with retries."""
    for attempt in range(max_retries):
        try:
            if url_or_path.startswith('http'):
                response = requests.get(url_or_path, timeout=10)
                img = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                img = Image.open(url_or_path).convert('RGB')
            return img
        except Exception as e:
            if attempt == max_retries - 1:
                return None
    return None


def get_binary_choice_ids(tokenizer):
    """Get token IDs for True/False."""
    true_id = tokenizer.encode(" True", add_special_tokens=False)[0]
    false_id = tokenizer.encode(" False", add_special_tokens=False)[0]
    return [true_id, false_id]


def train_and_eval_vsr(model, processor, arch, tokenizer, ds,
                       inject_layer, n_rounds, total_steps, seed,
                       grad_accum, tag, results_dir):
    """Train deliberation controller on VSR and evaluate."""
    random.seed(seed)
    torch.manual_seed(seed)

    n_layers = arch['n_layers']
    tapped = (n_layers // 4, n_layers // 2, 3 * n_layers // 4)

    controller = VisionMidLayerDeliberation(
        arch=arch, inject_layer=inject_layer, rank=64,
        d_state=512, n_slots=8, tapped_layers=tapped, topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    print(f'  Controller params: {sum(p.numel() for p in controller.parameters() if p.requires_grad)/1e6:.1f}M')

    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05
    )
    warmup = 200
    def lr_sched(s):
        if s < warmup: return s / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    choice_ids = get_binary_choice_ids(tokenizer)
    choice_ids_t = torch.tensor(choice_ids, device=device)

    # Prepare train/eval data
    train_data = list(ds['train'])
    eval_data = list(ds['test'])
    random.shuffle(train_data)

    t0 = time.time()
    losses = []
    skipped = 0
    optimizer.zero_grad(set_to_none=True)

    # Answer suffix
    answer_text = "\nAnswer:"
    answer_enc = tokenizer(answer_text, return_tensors='pt', add_special_tokens=False).to(device)
    with torch.no_grad():
        answer_emb = arch['embed_tokens'](answer_enc['input_ids'])

    print(f'  Training {total_steps} steps...', flush=True)

    for step in range(total_steps):
        sample = train_data[step % len(train_data)]

        # Load image from URL
        img = load_image_safe(sample.get('image_link', sample.get('image', '')))
        if img is None:
            skipped += 1
            continue

        caption = sample['caption']
        label = sample['label']  # 0 = False, 1 = True

        # Build prompt (LLaVA needs <image> token)
        if arch['type'] == 'llava':
            prompt = f"<image>\nIs this statement about the image true or false? \"{caption}\""
        else:
            prompt = f"Is this statement about the image true or false? \"{caption}\""

        try:
            with torch.no_grad():
                prompt_emb = get_fused_embeddings(model, processor, arch, img, prompt)

            label_t = torch.tensor([1 if label else 0], device=device, dtype=torch.long)  # True=0, False=1 in choice_ids

            all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_t, rounds=n_rounds)
            loss, lp = controller.compute_loss(all_cl, all_v, label_t)
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in controller.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(lp['final_ce'])

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                import traceback
                print(f'  WARNING step {step}: {e}', flush=True)
                traceback.print_exc()
            continue

        if (step + 1) % 500 == 0:
            avg = sum(losses[-500:]) / max(len(losses[-500:]), 1)
            print(f'  step {step+1}/{total_steps} | ce={avg:.4f} | skipped={skipped} | {time.time()-t0:.0f}s', flush=True)

    # === Eval ===
    print(f'\n  === VSR Eval ===', flush=True)
    controller.eval()
    results = {}

    for er in [3, 5]:
        correct = 0
        total_eval = 0
        eval_skip = 0

        for sample in eval_data:
            img = load_image_safe(sample.get('image_link', sample.get('image', '')))
            if img is None:
                eval_skip += 1
                continue

            caption = sample['caption']
            label = sample['label']
            if arch['type'] == 'llava':
                prompt = f"<image>\nIs this statement about the image true or false? \"{caption}\""
            else:
                prompt = f"Is this statement about the image true or false? \"{caption}\""

            try:
                with torch.no_grad():
                    prompt_emb = get_fused_embeddings(model, processor, arch, img, prompt)
                    all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_t, rounds=er)
                    pred = all_cl[-1].argmax(dim=-1).item()

                expected = 1 if label else 0
                if pred == expected:
                    correct += 1
                total_eval += 1
            except Exception:
                eval_skip += 1
                continue

        if total_eval > 0:
            acc = correct / total_eval
        else:
            acc = 0.0
        results[f'rounds={er}'] = {
            'accuracy': acc, 'correct': correct,
            'total': total_eval, 'skipped': eval_skip,
        }
        print(f'  rounds={er}: {acc:.4f} ({correct}/{total_eval}, {eval_skip} skipped)', flush=True)

    # === Baseline (no controller) ===
    print(f'  === Baseline (no controller) ===', flush=True)
    baseline_correct = 0
    baseline_total = 0
    for sample in eval_data[:200]:  # Quick baseline on first 200
        img = load_image_safe(sample.get('image_link', sample.get('image', '')))
        if img is None:
            continue
        caption = sample['caption']
        label = sample['label']
        if arch['type'] == 'llava':
            prompt = f"<image>\nIs this statement about the image true or false? \"{caption}\"\nAnswer:"
        else:
            prompt = f"Is this statement about the image true or false? \"{caption}\"\nAnswer:"

        try:
            with torch.no_grad():
                prompt_emb = get_fused_embeddings(model, processor, arch, img, prompt)
                T = prompt_emb.shape[1]
                pos_ids = torch.arange(T, device=device).unsqueeze(0)
                h = prompt_emb
                rotary_emb = arch['rotary_emb']
                if rotary_emb is not None:
                    pos_emb = rotary_emb(h, pos_ids)
                else:
                    pos_emb = None

                for layer in arch['layers']:
                    if pos_emb is not None:
                        out = layer(h, position_embeddings=pos_emb)
                    else:
                        out = layer(h)
                    h = out[0] if isinstance(out, tuple) else out
                h = arch['norm'](h)
                logits = arch['lm_head'](h)
                ans = logits[:, -1, choice_ids_t]
                pred = ans.argmax(dim=-1).item()

            expected = 1 if label else 0
            if pred == expected:
                baseline_correct += 1
            baseline_total += 1
        except Exception:
            continue

    if baseline_total > 0:
        baseline_acc = baseline_correct / baseline_total
    else:
        baseline_acc = 0.0
    results['baseline'] = {
        'accuracy': baseline_acc, 'correct': baseline_correct,
        'total': baseline_total,
    }
    print(f'  baseline: {baseline_acc:.4f} ({baseline_correct}/{baseline_total})', flush=True)

    # Save
    os.makedirs(results_dir, exist_ok=True)
    result_data = {
        'tag': tag, 'benchmark': 'vsr', 'model_type': arch['type'],
        'inject_layer': inject_layer, 'n_rounds': n_rounds,
        'total_steps': total_steps, 'seed': seed, 'grad_accum': grad_accum,
        'd_model': arch['d_model'], 'n_layers': arch['n_layers'],
        'train_skipped': skipped,
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(results_dir, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del controller, optimizer, scheduler
    torch.cuda.empty_cache()
    return result_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--benchmark', type=str, default='vsr',
                        choices=['vsr'])
    parser.add_argument('--inject_layer', type=int, default=12)
    parser.add_argument('--n_rounds', type=int, default=5)
    parser.add_argument('--total_steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    print(f'Loading {args.model}...', flush=True)

    from transformers import AutoProcessor
    if 'llava' in args.model.lower():
        from transformers import LlavaNextForConditionalGeneration
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16).to(device)
        processor = AutoProcessor.from_pretrained(args.model)
        tokenizer = processor.tokenizer
    elif 'paligemma' in args.model.lower():
        from transformers import PaliGemmaForConditionalGeneration
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16).to(device)
        processor = AutoProcessor.from_pretrained(args.model)
        tokenizer = processor.tokenizer
    elif 'llama' in args.model.lower() and 'vision' in args.model.lower():
        from transformers import MllamaForConditionalGeneration
        model = MllamaForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16).to(device)
        processor = AutoProcessor.from_pretrained(args.model)
        tokenizer = processor.tokenizer
    else:
        raise ValueError(f'Unknown model: {args.model}')

    for p in model.parameters():
        p.requires_grad = False
    print(f'Model loaded.', flush=True)

    arch = detect_vlm_architecture(model)
    print(f'  Architecture: {arch["type"]}, d_model={arch["d_model"]}, layers={arch["n_layers"]}', flush=True)

    # Model short name for tag
    if 'llava' in args.model.lower():
        model_short = 'llava7b'
    elif 'paligemma' in args.model.lower():
        model_short = 'paligemma10b'
    elif 'llama' in args.model.lower():
        model_short = 'llama11bv'
    else:
        model_short = 'vlm'

    if args.tag is None:
        args.tag = f'vision_{model_short}_{args.benchmark}_L{args.inject_layer}_{args.total_steps//1000}k_s{args.seed}'

    if args.benchmark == 'vsr':
        ds = load_vsr_dataset()
        train_and_eval_vsr(
            model, processor, arch, tokenizer, ds,
            inject_layer=args.inject_layer,
            n_rounds=args.n_rounds,
            total_steps=args.total_steps,
            seed=args.seed,
            grad_accum=args.grad_accum,
            tag=args.tag,
            results_dir=args.results_dir,
        )

    print('\n=== Vision deliberation complete ===', flush=True)


if __name__ == '__main__':
    main()
