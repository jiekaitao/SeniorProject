"""
Junction Fine-Tuning V4 — Adapter-Based Approach

Intellectual history (conversation with user, 2026-03-15):

  V3 used hidden-state MSE to push h_59 → h_49 (make the junction look like
  the first pass never happened). But this FIGHTS iterative refinement — in TRM,
  the whole point is that the second pass sees DIFFERENT input (F(h), not h).
  Making h_59 ≈ h_49 makes the second pass redundant.

  Training layer 60 directly doesn't work either — you'd just morph the copy of
  layer 50 back into the original layer 60, defeating the purpose of duplication.

  User's insight: insert a tiny adapter (bottleneck MLP) at the junction that
  adjusts the signal FORMAT without erasing the refinement CONTENT. The adapter
  acts as a "voltage converter" — the duplicated layers stay frozen (preserving
  iterative refinement), and the adapter learns minimal signal translation.

Architecture:
  JunctionAdapter = residual + bottleneck MLP
    h → h + Up(GELU(Down(h)))
    Down: hidden_dim → bottleneck_dim (e.g., 8192 → 256)
    Up: bottleneck_dim → hidden_dim (initialized near zero → starts as identity)

  Inserted after layer 59 (J1) and layer 69 (J2) in the 90-layer (50,60) model.

Loss:
  Logit KL with base model. We're not trying to beat the base model with the loss
  function — we're trying to RECOVER general capability. The improvement (if any)
  comes from the iterative refinement itself, unlocked by the adapter making the
  junction functional.

Parameter count:
  7B  (hidden=3584):  2 adapters × 2 × 3584 × 256 = ~3.7M params (0.04% of model)
  72B (hidden=8192):  2 adapters × 2 × 8192 × 256 = ~8.4M params (0.01% of model)
"""

import sys, os, json, copy, time, torch, gc, argparse
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe

RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/junction_ft_v4_adapter")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class JunctionAdapter(nn.Module):
    """
    Residual bottleneck adapter for junction signal translation.

    At initialization, up-projection is near-zero, so adapter(x) ≈ x (identity).
    Training learns minimal correction to translate the refined signal format.
    """
    def __init__(self, hidden_dim, bottleneck_dim=256, init_scale=0.01):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        # Near-zero init on up-projection → starts as identity (residual dominates)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=init_scale)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class AdapterWrappedLayer(nn.Module):
    """Wraps a transformer layer, applying an adapter AFTER its forward pass."""
    def __init__(self, original_layer, adapter):
        super().__init__()
        self.layer = original_layer
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        output = self.layer(*args, **kwargs)
        # Transformer layers return (hidden_states, ...) tuple
        if isinstance(output, tuple):
            h = output[0]
            h = self.adapter(h)
            return (h,) + output[1:]
        else:
            return self.adapter(output)


def run_adapter_ft(model_path, i, j, num_steps=200, bottleneck_dim=256,
                   lr=1e-3, tag="", save_model_path=None):
    """
    V4 adapter-based junction fine-tuning.

    For 7B: loads model, builds duplication, inserts adapters, trains.
    """
    print(f"\n{'='*60}")
    print(f"JUNCTION FT V4 (ADAPTER): {tag} — config ({i},{j})")
    print(f"{'='*60}")

    model, tokenizer = load_original_model(model_path)
    inner = model.model
    layers = list(inner.layers)
    N = len(layers)
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    dup_count = j - i

    # Training prompts — diverse mix
    prompts = [
        "The theory of general relativity states that",
        "In Python, a decorator is a function that",
        "To solve a quadratic equation, you can use",
        "Machine learning models are trained by",
        "The derivative of sin(x) is",
        "What is 78313 multiplied by 88537?",
        "The cube root of 74088 is approximately",
        "What is 9999 multiplied by 9999?",
        "The square root of 152399025 is",
        "What is 123456789 multiplied by 987654321?",
        "A linked list is a data structure where",
        "The speed of light in a vacuum is approximately",
        "The Pythagorean theorem states that",
        "To implement quicksort, you first choose a pivot",
        "The integral of e^x dx equals",
        "In economics, inflation is defined as",
    ]

    # Step 1: Baseline score
    print("\n--- Baseline (no duplication) ---")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    baseline = run_math_probe(gen_fn, verbose=False)
    print(f"  Baseline: {baseline['score']:.4f}")

    # Step 2: Collect teacher logits (for KL loss)
    print("\n--- Collecting teacher logits ---")
    teacher_data = []
    model.eval()
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt", truncation=True,
                       max_length=64).to(device)
        with torch.no_grad():
            out = model(**inp, use_cache=False)
        teacher_data.append({
            'input_ids': inp['input_ids'],
            'attention_mask': inp['attention_mask'],
            'logits': out.logits.detach().clone(),
        })

    # Step 3: Build duplicated model
    print(f"\n--- Building duplicated model ({i},{j}) ---")
    new_layers = layers[:j]
    for idx in range(i, j):
        new_layers.append(copy.deepcopy(layers[idx]))
    new_layers.extend(layers[j:])
    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    new_N = len(new_layers)

    # Step 4: Pre-adapter score
    print("\n--- Pre-adapter score ---")
    pre = run_math_probe(gen_fn, verbose=False)
    pre_delta = pre['score'] - baseline['score']
    print(f"  Pre-adapter: {pre['score']:.4f} (delta: {pre_delta:+.4f})")

    # Step 5: Insert adapters at junction points
    # J1: after layer j-1 (exit first pass → entry dup block)
    # J2: after layer j+dup_count-1 (exit dup block → resume original)
    j1_exit = j - 1
    j2_exit = j + dup_count - 1

    print(f"\n--- Inserting adapters ---")
    print(f"  Adapter 1: after layer {j1_exit} (before dup block entry at {j})")
    print(f"  Adapter 2: after layer {j2_exit} (before original resume at {j2_exit+1})")
    print(f"  Bottleneck: {hidden_dim} → {bottleneck_dim} → {hidden_dim}")

    adapter1 = JunctionAdapter(hidden_dim, bottleneck_dim).to(device).to(torch.bfloat16)
    adapter2 = JunctionAdapter(hidden_dim, bottleneck_dim).to(device).to(torch.bfloat16)

    # Wrap the exit layers with adapters
    inner.layers[j1_exit] = AdapterWrappedLayer(inner.layers[j1_exit], adapter1)
    inner.layers[j2_exit] = AdapterWrappedLayer(inner.layers[j2_exit], adapter2)

    adapter_params = list(adapter1.parameters()) + list(adapter2.parameters())
    trainable = sum(p.numel() for p in adapter_params)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Adapter params: {trainable:,} ({100*trainable/total:.4f}% of model)")

    # Step 6: Freeze everything except adapters
    for param in model.parameters():
        param.requires_grad = False
    for param in adapter_params:
        param.requires_grad = True

    # Step 7: Training — logit KL with base model
    optimizer = torch.optim.AdamW(adapter_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=num_steps,
        pct_start=0.1, anneal_strategy='cos',
        div_factor=10, final_div_factor=100
    )

    print(f"\n--- Training {num_steps} steps (logit KL loss) ---")
    model.train()
    # Only adapters are in train mode conceptually; rest is frozen
    losses = []

    for step in range(num_steps):
        total_loss = 0
        optimizer.zero_grad()

        for td in teacher_data:
            inp = {
                'input_ids': td['input_ids'],
                'attention_mask': td['attention_mask'],
            }
            student_out = model(**inp, use_cache=False)

            # KL divergence: student should match teacher's output distribution
            kl_loss = nn.functional.kl_div(
                nn.functional.log_softmax(student_out.logits[:, :-1, :].float(), dim=-1),
                nn.functional.softmax(td['logits'][:, :-1, :].float(), dim=-1),
                reduction='batchmean'
            )

            kl_loss.backward()
            total_loss += kl_loss.item()

            del student_out

        avg = total_loss / len(teacher_data)
        losses.append(avg)

        torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 25 == 0:
            print(f"    Step {step+1}/{num_steps} loss={avg:.6f}")

    # Step 8: Post-adapter evaluation
    print("\n--- Post-adapter score ---")
    model.eval()
    post = run_math_probe(gen_fn, verbose=False)
    post_delta = post['score'] - baseline['score']
    adapter_gain = post['score'] - pre['score']

    print(f"\n  {'='*50}")
    print(f"  V4 ADAPTER RESULTS: {tag}")
    print(f"  {'='*50}")
    print(f"  Baseline (no dup):     {baseline['score']:.4f}")
    print(f"  Pre-adapter (dup):     {pre['score']:.4f} ({pre_delta:+.4f})")
    print(f"  Post-adapter (dup+ad): {post['score']:.4f} ({post_delta:+.4f})")
    print(f"  Adapter gain:          {adapter_gain:+.4f}")
    print(f"  Adapter params:        {trainable:,} ({100*trainable/total:.4f}%)")
    if pre_delta > 0:
        preserved = post_delta / pre_delta * 100
        print(f"  Improvement preserved: {preserved:.1f}%")
    else:
        recovery = adapter_gain / abs(pre_delta) * 100 if abs(pre_delta) > 0 else 0
        print(f"  Quality recovery:      {recovery:.1f}%")

    result = {
        "tag": tag, "config": [i, j], "model": model_path,
        "bottleneck_dim": bottleneck_dim,
        "baseline": baseline['score'],
        "pre_adapter": pre['score'], "pre_delta": pre_delta,
        "post_adapter": post['score'], "post_delta": post_delta,
        "adapter_gain": adapter_gain,
        "adapter_params": trainable, "adapter_pct": 100 * trainable / total,
        "steps": num_steps, "lr": lr,
        "final_loss": losses[-1] if losses else 0,
        "losses": losses,
    }

    # Save adapter weights (tiny — just the adapters, not the whole model)
    adapter_save = {
        'adapter1': adapter1.state_dict(),
        'adapter2': adapter2.state_dict(),
        'config': {'i': i, 'j': j, 'hidden_dim': hidden_dim,
                   'bottleneck_dim': bottleneck_dim},
    }
    torch.save(adapter_save, RESULTS_DIR / f"adapter_weights_{tag}.pt")
    print(f"  Adapter weights saved ({trainable*2/1024/1024:.1f} MB)")

    # Optionally save full model
    if save_model_path:
        print(f"\n  Saving full model to {save_model_path}...")
        os.makedirs(save_model_path, exist_ok=True)
        model.save_pretrained(save_model_path, max_shard_size="5GB")
        tokenizer.save_pretrained(save_model_path)

    with open(RESULTS_DIR / f"results_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)

    del model, tokenizer, teacher_data
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_adapter_ft_72b_twostage(num_steps=100, bottleneck_dim=256, lr=1e-3):
    """
    Two-stage V4 for 72B: cache teacher logits from base model, then train
    adapters on saved (50,60) model.
    """
    BASE_MODEL = "/blue/cis4914/jietao/DeepPass/models/full/calme-2.1-qwen2-72b"
    DUP_MODEL = "/blue/cis4914/jietao/DeepPass/models/full/calme-2.1-qwen2-72b-dup-50-60"
    SAVE_PATH = "/blue/cis4914/jietao/DeepPass/models/full/calme-2.1-qwen2-72b-dup-50-60-adapter-v4"

    I, J = 50, 60
    DUP_COUNT = J - I
    CACHE_DIR = RESULTS_DIR / "cached_states_72b"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

    # ===== STAGE 1: Teacher data from base model =====
    print(f"\n{'='*60}")
    print("V4 72B — STAGE 1: Collecting teacher data")
    print(f"{'='*60}")

    model, tokenizer = load_original_model(BASE_MODEL)
    device = next(model.parameters()).device

    print("Collecting teacher logits...")
    teacher_data = []
    model.eval()
    for p in PROMPTS:
        inp = tokenizer(p, return_tensors="pt", truncation=True,
                       max_length=64).to(device)
        with torch.no_grad():
            out = model(**inp, use_cache=False)
        teacher_data.append({
            'input_ids': inp['input_ids'].cpu(),
            'attention_mask': inp['attention_mask'].cpu(),
            'logits': out.logits.cpu(),
        })
    torch.save(teacher_data, CACHE_DIR / "teacher_logits.pt")

    # Baseline
    print("Running baseline math probe...")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    baseline = run_math_probe(gen_fn, verbose=False)
    print(f"  Baseline: {baseline['score']:.4f}")
    with open(CACHE_DIR / "baseline.json", "w") as f:
        json.dump(baseline, f)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Stage 1 complete — base model unloaded\n")

    # ===== STAGE 2: Train adapters on saved (50,60) model =====
    print(f"\n{'='*60}")
    print("V4 72B — STAGE 2: Training adapters")
    print(f"{'='*60}")

    teacher_data = torch.load(CACHE_DIR / "teacher_logits.pt")
    with open(CACHE_DIR / "baseline.json") as f:
        baseline = json.load(f)

    model, tokenizer = load_original_model(DUP_MODEL)
    device = next(model.parameters()).device
    inner = model.model
    hidden_dim = model.config.hidden_size
    new_N = len(inner.layers)
    print(f"  Loaded: {new_N} layers, hidden_dim={hidden_dim}")

    # Pre-adapter math probe
    print("\nPre-adapter math probe...")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    pre = run_math_probe(gen_fn, verbose=False)
    pre_delta = pre['score'] - baseline['score']
    print(f"  Pre-adapter: {pre['score']:.4f} (delta: {pre_delta:+.4f})")

    # Junction points in saved 90-layer model
    j1_exit = J - 1          # 59
    j2_exit = J + DUP_COUNT - 1  # 69

    print(f"\nInserting adapters after layers {j1_exit} and {j2_exit}")
    print(f"  Bottleneck: {hidden_dim} → {bottleneck_dim} → {hidden_dim}")

    adapter1 = JunctionAdapter(hidden_dim, bottleneck_dim).to(device).to(torch.bfloat16)
    adapter2 = JunctionAdapter(hidden_dim, bottleneck_dim).to(device).to(torch.bfloat16)

    inner.layers[j1_exit] = AdapterWrappedLayer(inner.layers[j1_exit], adapter1)
    inner.layers[j2_exit] = AdapterWrappedLayer(inner.layers[j2_exit], adapter2)

    # Freeze everything except adapters
    for param in model.parameters():
        param.requires_grad = False
    adapter_params = list(adapter1.parameters()) + list(adapter2.parameters())
    for param in adapter_params:
        param.requires_grad = True

    trainable = sum(p.numel() for p in adapter_params)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Adapter params: {trainable:,} ({100*trainable/total:.4f}%)")

    # Training
    optimizer = torch.optim.AdamW(adapter_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=num_steps,
        pct_start=0.1, anneal_strategy='cos',
        div_factor=10, final_div_factor=100
    )

    print(f"\n--- Training {num_steps} steps ---")
    model.train()
    losses = []

    for step in range(num_steps):
        total_loss = 0
        optimizer.zero_grad()

        for td in teacher_data:
            inp = {
                'input_ids': td['input_ids'].to(device),
                'attention_mask': td['attention_mask'].to(device),
            }
            teacher_logits = td['logits'].to(device)

            student_out = model(**inp, use_cache=False)

            kl_loss = nn.functional.kl_div(
                nn.functional.log_softmax(student_out.logits[:, :-1, :].float(), dim=-1),
                nn.functional.softmax(teacher_logits[:, :-1, :].float(), dim=-1),
                reduction='batchmean'
            )

            kl_loss.backward()
            total_loss += kl_loss.item()

            del student_out, teacher_logits
            torch.cuda.empty_cache()

        avg = total_loss / len(teacher_data)
        losses.append(avg)

        torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 10 == 0:
            print(f"    Step {step+1}/{num_steps} loss={avg:.6f}")

    # Post-adapter evaluation
    print("\n--- Post-adapter math probe ---")
    model.eval()
    post = run_math_probe(gen_fn, verbose=False)
    post_delta = post['score'] - baseline['score']
    adapter_gain = post['score'] - pre['score']

    print(f"\n  {'='*50}")
    print(f"  V4 ADAPTER 72B RESULTS")
    print(f"  {'='*50}")
    print(f"  Baseline (no dup):     {baseline['score']:.4f}")
    print(f"  Pre-adapter (dup):     {pre['score']:.4f} ({pre_delta:+.4f})")
    print(f"  Post-adapter (dup+ad): {post['score']:.4f} ({post_delta:+.4f})")
    print(f"  Adapter gain:          {adapter_gain:+.4f}")
    print(f"  Adapter params:        {trainable:,} ({100*trainable/total:.4f}%)")

    result = {
        "tag": "72b_50_60_adapter", "config": [I, J],
        "bottleneck_dim": bottleneck_dim,
        "baseline": baseline['score'],
        "pre_adapter": pre['score'], "pre_delta": pre_delta,
        "post_adapter": post['score'], "post_delta": post_delta,
        "adapter_gain": adapter_gain,
        "adapter_params": trainable, "adapter_pct": 100 * trainable / total,
        "steps": num_steps, "lr": lr,
        "final_loss": losses[-1] if losses else 0,
        "losses": losses,
    }

    # Save adapter weights
    adapter_save = {
        'adapter1': adapter1.state_dict(),
        'adapter2': adapter2.state_dict(),
        'config': {'i': I, 'j': J, 'hidden_dim': hidden_dim,
                   'bottleneck_dim': bottleneck_dim},
    }
    torch.save(adapter_save, RESULTS_DIR / "adapter_weights_72b.pt")

    # Save full model
    print(f"\nSaving model to {SAVE_PATH}...")
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH, max_shard_size="5GB")
    tokenizer.save_pretrained(SAVE_PATH)

    with open(RESULTS_DIR / "results_72b.json", "w") as f:
        json.dump(result, f, indent=2)

    del model, tokenizer, teacher_data
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["7b", "72b"], default="7b")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.mode == "72b":
        run_adapter_ft_72b_twostage(
            num_steps=args.steps, bottleneck_dim=args.bottleneck, lr=args.lr
        )
    else:
        # 7B experiments — test on good and bad configs
        M7B = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"
        results = []

        # Good config
        results.append(run_adapter_ft(
            M7B, 10, 11, num_steps=args.steps, bottleneck_dim=args.bottleneck,
            lr=args.lr, tag="7b_good_10_11"
        ))

        # Another good config
        results.append(run_adapter_ft(
            M7B, 18, 21, num_steps=args.steps, bottleneck_dim=args.bottleneck,
            lr=args.lr, tag="7b_good_18_21"
        ))

        # Bad config
        results.append(run_adapter_ft(
            M7B, 4, 9, num_steps=args.steps, bottleneck_dim=args.bottleneck,
            lr=args.lr, tag="7b_bad_4_9"
        ))

        # Summary
        print(f"\n{'='*80}")
        print("V4 ADAPTER — SUMMARY")
        print(f"{'='*80}")
        print(f"{'Tag':>20} {'Baseline':>9} {'Pre':>9} {'Post':>9} {'Gain':>9} {'Params':>10}")
        for r in results:
            print(f"{r['tag']:>20} {r['baseline']:9.4f} {r['pre_adapter']:9.4f} "
                  f"{r['post_adapter']:9.4f} {r['adapter_gain']:+9.4f} {r['adapter_params']:>10,}")

        with open(RESULTS_DIR / "results_7b_all.json", "w") as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
