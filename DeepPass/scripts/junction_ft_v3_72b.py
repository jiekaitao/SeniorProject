"""
Junction Fine-Tuning V3 — 72B Memory-Efficient Version

Two-stage approach to fit within B200 192GB:
  Stage 1: Load base model → compute teacher hidden states → cache to disk → unload
  Stage 2: Load saved (50,60) model → train 4 junction layers → evaluate → save model
"""

import sys, os, json, time, torch, gc
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe

BASE_MODEL = "/blue/cis4914/jietao/DeepPass/models/full/calme-2.1-qwen2-72b"
DUP_MODEL = "/blue/cis4914/jietao/DeepPass/models/full/calme-2.1-qwen2-72b-dup-50-60"
SAVE_PATH = "/blue/cis4914/jietao/DeepPass/models/full/calme-2.1-qwen2-72b-dup-50-60-jft-v3"

RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/junction_ft_v3_72b")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RESULTS_DIR / "cached_states"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Config: layers 50-59 duplicated → 90-layer model
# In saved model: layers 0-49 (orig), 50-59 (orig first pass), 60-69 (dup second pass), 70-89 (orig 60-79)
I, J = 50, 60
DUP_COUNT = J - I  # 10

# Junction points in the 90-layer saved model:
J1_EXIT  = J - 1          # 59: end of first pass
J1_ENTRY = J              # 60: start of dup block (copy of layer 50)
J2_EXIT  = J + DUP_COUNT - 1  # 69: end of dup block
J2_ENTRY = J + DUP_COUNT      # 70: resume original flow (originally layer 60)

# Teacher target: output of layer I-1 (49) in base model = what layer 50 expects as input
TEACHER_LAYER = I - 1  # 49

# Reduced set for 72B — 8 prompts, 100 steps (vs 24/200 on 7B)
PROMPTS = [
    "The theory of general relativity states that",
    "In Python, a decorator is a function that",
    "To solve a quadratic equation, you can use",
    "What is 78313 multiplied by 88537?",
    "The cube root of 74088 is approximately",
    "What is 9999 multiplied by 9999?",
    "A linked list is a data structure where",
    "The Pythagorean theorem states that",
]
NUM_STEPS = 100


def collect_hidden_states_at_layer(model, tokenizer, prompts, layer_idx, device):
    """Collect hidden states at a specific layer using hooks."""
    all_states = []

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        all_states.append(h.detach().cpu())  # Save to CPU immediately

    inner = model.model
    hook_handle = inner.layers[layer_idx].register_forward_hook(hook_fn)

    for p in prompts:
        inp = tokenizer(p, return_tensors="pt", truncation=True,
                       max_length=64).to(device)
        with torch.no_grad():
            model(**inp, use_cache=False)

    hook_handle.remove()
    return all_states


def stage1_collect_teacher_data():
    """Load base model, collect teacher hidden states and logits, cache to disk, unload."""
    print(f"\n{'='*60}")
    print("STAGE 1: Collecting teacher data from base model")
    print(f"{'='*60}")

    model, tokenizer = load_original_model(BASE_MODEL)
    device = next(model.parameters()).device

    # Collect teacher hidden states at layer 49 (what layer 50 expects)
    print(f"Collecting hidden states at layer {TEACHER_LAYER}...")
    teacher_states = collect_hidden_states_at_layer(
        model, tokenizer, PROMPTS, TEACHER_LAYER, device
    )
    torch.save(teacher_states, CACHE_DIR / "teacher_hidden_states.pt")
    print(f"  Saved {len(teacher_states)} hidden state tensors")

    # Collect teacher logits (needed if config turns out to be "bad")
    print("Collecting teacher logits...")
    teacher_data = []
    for p in PROMPTS:
        inp = tokenizer(p, return_tensors="pt", truncation=True,
                       max_length=64).to(device)
        with torch.no_grad():
            out = model(**inp, use_cache=False)
        teacher_data.append({
            'input_ids': inp['input_ids'].cpu(),
            'attention_mask': inp['attention_mask'].cpu(),
            'logits': out.logits.cpu()
        })
    torch.save(teacher_data, CACHE_DIR / "teacher_logits.pt")
    print(f"  Saved logits for {len(teacher_data)} prompts")

    # Baseline math probe
    print("\nRunning baseline math probe...")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    baseline = run_math_probe(gen_fn, verbose=False)
    print(f"  Baseline score: {baseline['score']:.4f}")
    with open(CACHE_DIR / "baseline.json", "w") as f:
        json.dump(baseline, f)

    # Unload
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Stage 1 complete — base model unloaded\n")
    return baseline


def stage2_train_and_evaluate():
    """Load saved (50,60) model, train junction layers, evaluate, save."""
    print(f"\n{'='*60}")
    print("STAGE 2: Training junction layers on (50,60) model")
    print(f"{'='*60}")

    # Load cached data
    teacher_states = torch.load(CACHE_DIR / "teacher_hidden_states.pt")
    teacher_data = torch.load(CACHE_DIR / "teacher_logits.pt")
    with open(CACHE_DIR / "baseline.json") as f:
        baseline = json.load(f)

    # Load duplicated model
    print(f"Loading saved (50,60) model from {DUP_MODEL}...")
    model, tokenizer = load_original_model(DUP_MODEL)
    device = next(model.parameters()).device
    inner = model.model
    new_N = len(inner.layers)
    print(f"  Model loaded: {new_N} layers")

    # Pre-FT math probe
    print("\nRunning pre-FT math probe...")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    pre = run_math_probe(gen_fn, verbose=False)
    pre_delta = pre['score'] - baseline['score']
    is_good_config = pre_delta > 0
    print(f"  Pre-FT: {pre['score']:.4f} (delta: {pre_delta:+.4f}) — {'GOOD' if is_good_config else 'BAD'}")

    # Identify junction layers
    junction_layers = sorted(set([J1_EXIT, J1_ENTRY, J2_EXIT, J2_ENTRY]))
    junction_layers = [jl for jl in junction_layers if jl < new_N]
    print(f"  Junction layers: {junction_layers}")
    print(f"  J1: {J1_EXIT}→{J1_ENTRY}, J2: {J2_EXIT}→{J2_ENTRY}")

    # Freeze everything except junction layers
    for param in model.parameters():
        param.requires_grad = False

    lr_map = {J1_EXIT: 3e-5, J1_ENTRY: 2e-5, J2_EXIT: 1e-5, J2_ENTRY: 5e-6}
    param_groups = []
    trainable = 0
    for jl in junction_layers:
        lr = lr_map.get(jl, 1e-5)
        params = list(inner.layers[jl].parameters())
        for p in params:
            p.requires_grad = True
            trainable += p.numel()
        param_groups.append({"params": params, "lr": lr})

    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    # Hook at J1_EXIT (layer 59) to capture hidden states entering the dup block
    junction_hidden_states = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            junction_hidden_states[name] = h
        return hook_fn

    hook_j1 = inner.layers[J1_EXIT].register_forward_hook(make_hook('j1_output'))

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g['lr'] for g in param_groups],
        total_steps=NUM_STEPS, pct_start=0.1,
        anneal_strategy='cos', div_factor=10, final_div_factor=100
    )

    # Training
    mode = "GOOD CONFIG — hidden-state MSE only" if is_good_config else "BAD CONFIG — MSE + logit KL"
    print(f"\n--- Training {NUM_STEPS} steps ({mode}) ---")
    model.train()
    losses = []

    for step in range(NUM_STEPS):
        total_loss = 0
        optimizer.zero_grad()

        for idx in range(len(PROMPTS)):
            td = teacher_data[idx]
            inp = {
                'input_ids': td['input_ids'].to(device),
                'attention_mask': td['attention_mask'].to(device),
            }
            junction_hidden_states.clear()

            # Forward pass
            student_out = model(**inp, use_cache=False)

            # Loss 1: Hidden-state MSE at junction
            if 'j1_output' in junction_hidden_states and idx < len(teacher_states):
                h_student = junction_hidden_states['j1_output']
                h_teacher = teacher_states[idx].to(device)
                min_len = min(h_student.shape[1], h_teacher.shape[1])
                h_s = h_student[:, :min_len, :]
                h_t = h_teacher[:, :min_len, :]
                hidden_loss = nn.functional.mse_loss(h_s, h_t)
            else:
                hidden_loss = torch.tensor(0.0, device=device)

            if is_good_config:
                loss = hidden_loss
            else:
                teacher_logits = td['logits'].to(device)
                kl_loss = nn.functional.kl_div(
                    nn.functional.log_softmax(student_out.logits[:, :-1, :], dim=-1),
                    nn.functional.softmax(teacher_logits[:, :-1, :], dim=-1),
                    reduction='batchmean'
                )
                loss = 0.7 * hidden_loss + 0.3 * kl_loss
                del teacher_logits

            if loss.requires_grad:
                loss.backward()
            total_loss += loss.item()

            del student_out
            if not is_good_config:
                pass  # logits already deleted
            torch.cuda.empty_cache()

        avg = total_loss / len(PROMPTS)
        losses.append(avg)

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()

        if (step + 1) % 10 == 0:
            print(f"    Step {step+1}/{NUM_STEPS} loss={avg:.6f}")

    hook_j1.remove()

    # Post-FT evaluation
    print("\n--- Post-FT math probe ---")
    model.eval()
    post = run_math_probe(gen_fn, verbose=False)
    post_delta = post['score'] - baseline['score']
    ft_gain = post['score'] - pre['score']

    print(f"\n  {'='*50}")
    print(f"  V3 72B RESULTS: config ({I},{J})")
    print(f"  {'='*50}")
    print(f"  Baseline (no dup): {baseline['score']:.4f}")
    print(f"  Pre-FT (dup):      {pre['score']:.4f} ({pre_delta:+.4f})")
    print(f"  Post-FT (dup+jft): {post['score']:.4f} ({post_delta:+.4f})")
    print(f"  FT gain:           {ft_gain:+.4f}")
    print(f"  Config type:       {'GOOD' if is_good_config else 'BAD'}")
    if is_good_config:
        preserved = "YES" if post_delta >= pre_delta * 0.9 else ("PARTIALLY" if post_delta > 0 else "NO")
        print(f"  Improvement preserved: {preserved}")
    else:
        recovery = ft_gain / abs(pre_delta) * 100 if abs(pre_delta) > 0 else 0
        print(f"  Recovery: {recovery:.1f}% of lost quality")

    # Save results
    result = {
        "config": [I, J], "model": DUP_MODEL,
        "baseline": baseline['score'],
        "pre_ft": pre['score'], "pre_delta": pre_delta,
        "post_ft": post['score'], "post_delta": post_delta,
        "ft_gain": ft_gain, "is_good_config": is_good_config,
        "junction_layers": junction_layers,
        "trainable": trainable, "trainable_pct": 100 * trainable / total,
        "steps": NUM_STEPS, "final_loss": losses[-1] if losses else 0,
        "losses": losses,
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

    # Save model for lm-eval
    print(f"\nSaving fine-tuned model to {SAVE_PATH}...")
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH, max_shard_size="5GB")
    tokenizer.save_pretrained(SAVE_PATH)
    print("Model saved!")

    del model, tokenizer, teacher_states, teacher_data
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    t0 = time.time()

    # Stage 1: teacher data from base model
    baseline = stage1_collect_teacher_data()

    # Stage 2: train on duplicated model
    result = stage2_train_and_evaluate()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
