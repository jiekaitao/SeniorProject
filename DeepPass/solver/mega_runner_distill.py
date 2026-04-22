"""LoRA -> Controller distillation.

Test of GPT-5.4 Pro's H3 hypothesis: is the controller fundamentally unable
to modify semantic associations, or can it learn them if given a teacher?

Protocol:
  1. Train a LoRA on the target task (e.g., WinoGrande) — teacher.
  2. Run teacher over train set, cache logit distribution per sample.
  3. Train a fresh controller with loss = alpha*CE(controller, y) + (1-alpha)*KL(controller, teacher).
  4. Compare to pure-CE controller baseline (which we already have).

If distilled controller reaches LoRA's accuracy, controller CAN learn semantic
associations — it was just missing the right training signal. This would
invalidate H3 and open a path to universal performance.

If distilled controller plateaus at controller-only level, H3 is confirmed —
the architecture can't express the LoRA solution even with supervision.
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from mega_runner_benchmarks import (
    FlexMidLayerDeliberation, BENCHMARK_LOADERS,
)

device = torch.device('cuda')


def get_choice_tokens(tokenizer, n_choices):
    letters = ['A', 'B', 'C', 'D', 'E'][:n_choices]
    return [tokenizer.encode(f' {c}', add_special_tokens=False)[0] for c in letters]


def train_lora_teacher(base_model, tokenizer, data, n_choices, total_steps, seed,
                       grad_accum, lr=1e-4, rank=64):
    """Train a LoRA on the task as teacher."""
    random.seed(seed); torch.manual_seed(seed)
    choice_ids = get_choice_tokens(tokenizer, n_choices)
    choice_ids_t = torch.tensor(choice_ids, device=device)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=rank * 2,
        lora_dropout=0.05, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
    )
    teacher = get_peft_model(base_model, config)
    trainable_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    print(f'  LoRA teacher: {trainable_params/1e6:.1f}M trainable', flush=True)

    optimizer = torch.optim.AdamW([p for p in teacher.parameters() if p.requires_grad],
                                    lr=lr, weight_decay=0.05)
    warmup = 200
    def lr_sched(s):
        if s < warmup: return s / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    teacher.train()
    train_data = data['train']
    random.shuffle(train_data)
    t0 = time.time()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(total_steps):
        sample = train_data[step % len(train_data)]
        text = sample['text'][:1500] + '\nAnswer:'
        label = sample['label']
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
        out = teacher(enc['input_ids'])
        logits = out.logits[:, -1, choice_ids_t]
        label_t = torch.tensor([label], device=device, dtype=torch.long)
        loss = F.cross_entropy(logits.float(), label_t) / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in teacher.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item() * grad_accum)
        if (step + 1) % 1000 == 0:
            avg = sum(losses[-1000:]) / 1000
            print(f'  teacher step {step+1} | ce={avg:.4f} | {time.time()-t0:.0f}s', flush=True)
    return teacher, choice_ids_t


def cache_teacher_logits(teacher, tokenizer, train_data, choice_ids_t, n_cache=4000):
    """Run teacher over train set, cache softmax probabilities per sample."""
    teacher.eval()
    cached = []
    print(f'  Caching teacher logits for {n_cache} train samples...', flush=True)
    with torch.no_grad():
        for i, sample in enumerate(train_data[:n_cache]):
            text = sample['text'][:1500] + '\nAnswer:'
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
            out = teacher(enc['input_ids'])
            logits = out.logits[:, -1, choice_ids_t]
            probs = F.softmax(logits.squeeze(0).float(), dim=-1).cpu()
            cached.append(probs)
    return cached  # list of (n_choices,) tensors


def eval_lora(teacher, tokenizer, test_data, choice_ids_t, limit=500):
    teacher.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in test_data[:limit]:
            text = sample['text'][:1500] + '\nAnswer:'
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
            out = teacher(enc['input_ids'])
            logits = out.logits[:, -1, choice_ids_t]
            pred = logits.argmax(dim=-1).item()
            if pred == sample['label']:
                correct += 1
            total += 1
    return correct / max(total, 1), correct, total


def train_distilled_controller(base_model, tokenizer, data, cached_probs, n_choices,
                                  total_steps, seed, grad_accum,
                                  kl_weight=0.7, inject_layer=12, n_rounds=5):
    """Train controller with distillation + CE from LoRA-teacher logits."""
    random.seed(seed); torch.manual_seed(seed)
    controller = FlexMidLayerDeliberation(
        frozen_llm=base_model, inject_layer=inject_layer, n_choices=n_choices,
        rank=64, d_state=512, n_slots=8, tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in controller.parameters() if p.requires_grad)
    print(f'  Controller: {n_params/1e6:.1f}M trainable', flush=True)

    optimizer = torch.optim.AdamW([p for p in controller.parameters() if p.requires_grad],
                                    lr=1e-4, weight_decay=0.05)
    warmup = 200
    def lr_sched(s):
        if s < warmup: return s / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    choice_ids = get_choice_tokens(tokenizer, n_choices)
    choice_ids_t = torch.tensor(choice_ids, device=device)

    train_data = data['train']
    lm_model = base_model.model
    answer_enc = tokenizer('\nAnswer:', return_tensors='pt', add_special_tokens=False).to(device)
    with torch.no_grad():
        answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])

    # Shuffle indices such that cached logits are aligned with train_data[:n_cache]
    n_cache = len(cached_probs)
    idx_pool = list(range(n_cache))
    random.shuffle(idx_pool)

    t0 = time.time()
    losses = []
    kl_losses = []
    ce_losses = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(total_steps):
        idx = idx_pool[step % len(idx_pool)]
        sample = train_data[idx]
        teacher_probs = cached_probs[idx].to(device).to(torch.float32)
        text = sample['text'][:1500]
        label = sample['label']

        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(enc['input_ids'])
        label_t = torch.tensor([label], device=device, dtype=torch.long)

        all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_t, rounds=n_rounds)
        final = all_cl[-1]

        # CE loss
        ce = F.cross_entropy(final.float(), label_t)
        # KL loss: KL(student || teacher) where student uses log-softmax
        student_log = F.log_softmax(final.float(), dim=-1)
        kl = F.kl_div(student_log, teacher_probs.unsqueeze(0), reduction='batchmean')
        # Weighted combined loss
        total_loss = (1 - kl_weight) * ce + kl_weight * kl

        # Plus the base controller losses (verifier + progress) at small weight
        _, parts = controller.compute_loss(all_cl, all_v, label_t)
        # Extract just the verifier and progress parts (not the CE)
        vl = torch.tensor(parts['verify_loss'], device=device)
        pl = torch.tensor(parts['progress_loss'], device=device)
        total_loss = total_loss + 0.5 * vl + 0.1 * pl

        total_loss = total_loss / grad_accum
        total_loss.backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in controller.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(total_loss.item() * grad_accum)
        kl_losses.append(kl.item())
        ce_losses.append(ce.item())
        if (step + 1) % 500 == 0:
            avg_kl = sum(kl_losses[-500:]) / 500
            avg_ce = sum(ce_losses[-500:]) / 500
            print(f'  distill step {step+1} | ce={avg_ce:.4f} kl={avg_kl:.4f} | {time.time()-t0:.0f}s', flush=True)
    return controller, choice_ids_t


def eval_controller(controller, base_model, tokenizer, test_data, choice_ids_t,
                    eval_rounds, limit=500):
    controller.eval()
    correct = 0
    total = 0
    lm_model = base_model.model
    answer_enc = tokenizer('\nAnswer:', return_tensors='pt', add_special_tokens=False).to(device)
    with torch.no_grad():
        answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
        for sample in test_data[:limit]:
            text = sample['text'][:1500]
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
            prompt_emb = lm_model.embed_tokens(enc['input_ids'])
            all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_t, rounds=eval_rounds)
            pred = all_cl[-1].argmax(dim=-1).item()
            if pred == sample['label']:
                correct += 1
            total += 1
    return correct / max(total, 1), correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/full/Llama-3.1-8B-Instruct')
    parser.add_argument('--benchmark', type=str, default='winogrande')
    parser.add_argument('--teacher_steps', type=int, default=6000)
    parser.add_argument('--student_steps', type=int, default=8000)
    parser.add_argument('--n_cache', type=int, default=4000)
    parser.add_argument('--kl_weight', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--results_dir', type=str, default='results/data/distill')
    args = parser.parse_args()

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    print('Model loaded.', flush=True)

    print(f'Loading benchmark {args.benchmark}...', flush=True)
    data, n_choices = BENCHMARK_LOADERS[args.benchmark]()

    # Baseline (zero-shot) on test
    choice_ids = get_choice_tokens(tokenizer, n_choices)
    choice_ids_t = torch.tensor(choice_ids, device=device)
    print('\n=== Baseline eval ===', flush=True)
    baseline_acc, bl_c, bl_t = eval_lora(base_model, tokenizer, data['test'], choice_ids_t, limit=500)
    print(f'Baseline: {baseline_acc:.4f} ({bl_c}/{bl_t})', flush=True)

    # Train LoRA teacher
    print('\n=== Stage 1: Train LoRA teacher ===', flush=True)
    teacher, _ = train_lora_teacher(base_model, tokenizer, data, n_choices,
                                      args.teacher_steps, args.seed, args.grad_accum)

    # Eval teacher
    print('\n=== Teacher eval ===', flush=True)
    teacher_acc, t_c, t_t = eval_lora(teacher, tokenizer, data['test'], choice_ids_t, limit=500)
    print(f'Teacher (LoRA): {teacher_acc:.4f} ({t_c}/{t_t})', flush=True)

    # Cache teacher logits
    cached = cache_teacher_logits(teacher, tokenizer, data['train'], choice_ids_t, n_cache=args.n_cache)

    # Unload teacher to free GPU
    print('\n=== Unloading teacher ===', flush=True)
    base_model_unloaded = teacher.unload()
    for p in base_model_unloaded.parameters(): p.requires_grad = False
    torch.cuda.empty_cache()

    # Train distilled controller
    print('\n=== Stage 2: Distill to controller ===', flush=True)
    controller, _ = train_distilled_controller(
        base_model_unloaded, tokenizer, data, cached, n_choices,
        total_steps=args.student_steps, seed=args.seed, grad_accum=args.grad_accum,
        kl_weight=args.kl_weight)

    # Eval controller
    print('\n=== Controller eval ===', flush=True)
    results = {'baseline': {'accuracy': baseline_acc, 'correct': bl_c, 'total': bl_t},
               'teacher': {'accuracy': teacher_acc, 'correct': t_c, 'total': t_t}}
    for er in [3, 5, 8]:
        acc, c, t = eval_controller(controller, base_model_unloaded, tokenizer,
                                      data['test'], choice_ids_t, eval_rounds=er, limit=500)
        results[f'rounds={er}'] = {'accuracy': acc, 'correct': c, 'total': t}
        print(f'  rounds={er}: {acc:.4f} ({c}/{t})', flush=True)

    # Save
    model_short = 'inst' if 'instruct' in args.model.lower() else 'base'
    tag = f'distill_{model_short}_{args.benchmark}_kl{args.kl_weight}_t{args.teacher_steps//1000}k_s{args.student_steps//1000}k_seed{args.seed}'
    os.makedirs(args.results_dir, exist_ok=True)
    out = {'tag': tag, 'model': args.model, 'benchmark': args.benchmark,
           'teacher_steps': args.teacher_steps, 'student_steps': args.student_steps,
           'kl_weight': args.kl_weight, 'seed': args.seed, 'grad_accum': args.grad_accum,
           'results': results, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(os.path.join(args.results_dir, f'{tag}.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {tag}.json', flush=True)


if __name__ == '__main__':
    main()
