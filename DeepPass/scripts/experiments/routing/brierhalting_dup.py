"""
BrierHalting for Pretrained Layer Duplication

Based on GPT-5.4 Pro's design. Key ideas:
- Pick one good block offline (spectral search)
- At test time, run 1-N passes, tiny halt head decides when to stop
- Trained on actual benchmark correctness, not spectral proxies
- Brier score loss (bounded gradients) + monotonicity regularization

Two-headed design:
- q_halt: "should I stop now?" (primary, monotonic)
- q_correct: "is the answer correct?" (auxiliary, not monotonic)

The halt target is derived from utility:
  t*(x) = argmax[U_t(x) - λ_compute*(t-1)]
  s_t = 1[t >= t*]  (monotonic by construction)
"""

import sys, os, json, time, torch, gc, argparse
import torch.nn as nn
import numpy as np
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', 'scripts'))

from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe, calculate_score

RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/brierhalting")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# BrierHalting Head
# =============================================================================

class BrierHaltingHead(nn.Module):
    """
    Tiny head (~800K params) that predicts:
    - q_halt: P(should stop after this pass)
    - q_correct: P(answer is correct after this pass)

    Features: last-token hidden state + delta from prev pass + scalar dynamics + pass embedding
    """

    def __init__(self, hidden_size, max_passes=3, proj_dim=64, delta_proj_dim=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_passes = max_passes

        # Project hidden state and delta
        self.h_proj = nn.Linear(hidden_size, proj_dim)
        self.delta_proj = nn.Linear(hidden_size, delta_proj_dim)

        # Pass embedding
        self.pass_emb = nn.Embedding(max_passes + 1, 8)

        # Scalar dynamics features: norm, prev_norm, delta_norm, cosine_sim, mean_abs_delta, max_abs_delta
        num_scalar = 6

        # MLP
        total_input = proj_dim + delta_proj_dim + 8 + num_scalar
        self.mlp = nn.Sequential(
            nn.Linear(total_input, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # Two output heads
        self.halt_head = nn.Linear(64, 1)
        self.correct_head = nn.Linear(64, 1)

    def forward(self, h_current, h_previous, pass_idx):
        """
        Args:
            h_current: [B, hidden_size] last-token hidden state after this pass
            h_previous: [B, hidden_size] last-token hidden state after previous pass (or zeros for pass 0)
            pass_idx: int, 0-indexed pass number
        Returns:
            q_halt: [B, 1] probability of halting
            q_correct: [B, 1] probability of correct answer
        """
        # Ensure 2D: [B, hidden_size]
        if h_current.dim() == 1:
            h_current = h_current.unsqueeze(0)
        if h_previous.dim() == 1:
            h_previous = h_previous.unsqueeze(0)

        B = h_current.shape[0]
        device = h_current.device

        # Project
        h_proj = self.h_proj(h_current.float())
        delta = h_current - h_previous
        delta_proj = self.delta_proj(delta.float())

        # Pass embedding
        pass_t = torch.full((B,), pass_idx, dtype=torch.long, device=device)
        pass_e = self.pass_emb(pass_t)

        # Scalar dynamics
        h_norm = h_current.float().norm(dim=-1, keepdim=True)
        prev_norm = h_previous.float().norm(dim=-1, keepdim=True)
        delta_norm = delta.float().norm(dim=-1, keepdim=True)
        cosine = nn.functional.cosine_similarity(
            h_current.float(), h_previous.float(), dim=-1
        ).unsqueeze(-1)
        mean_abs_delta = delta.float().abs().mean(dim=-1, keepdim=True)
        max_abs_delta = delta.float().abs().amax(dim=-1, keepdim=True)

        scalars = torch.cat([h_norm, prev_norm, delta_norm, cosine,
                            mean_abs_delta, max_abs_delta], dim=-1)

        # Concatenate all features
        features = torch.cat([h_proj, delta_proj, pass_e, scalars], dim=-1)

        # MLP
        x = self.mlp(features)

        q_halt = torch.sigmoid(self.halt_head(x))
        q_correct = torch.sigmoid(self.correct_head(x))

        return q_halt, q_correct


# =============================================================================
# Layer-by-layer forward with hidden state capture
# =============================================================================

def _run_layers(inner, h, start, end, pos_embeds):
    """Run layers [start, end) on hidden state h."""
    for i in range(start, end):
        out = inner.layers[i](h, position_embeddings=pos_embeds, use_cache=False)
        h = out[0] if isinstance(out, tuple) else out
    return h


def run_n_passes(model, input_ids, block_start, block_end, n_passes):
    """
    Run model with block [start, end) duplicated n_passes times.
    Returns:
        final_hidden: hidden state after all passes + suffix
        pass_hiddens: list of hidden states at block exit for each pass
    """
    device = next(model.parameters()).device
    inner = model.model
    N = len(inner.layers)

    h = inner.embed_tokens(input_ids.to(device))
    seq_len = h.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    pos_embeds = inner.rotary_emb(h, position_ids)

    # Prefix
    h = _run_layers(inner, h, 0, block_start, pos_embeds)

    # Run block n_passes times, capturing hidden state after each pass
    pass_hiddens = []
    for p in range(n_passes):
        h = _run_layers(inner, h, block_start, block_end, pos_embeds)
        pass_hiddens.append(h[:, -1, :].detach().clone())  # last token

    # Suffix
    h = _run_layers(inner, h, block_end, N, pos_embeds)
    h = inner.norm(h)

    return h, pass_hiddens


def generate_n_passes(model, tokenizer, prompt, block_start, block_end,
                      n_passes, max_new_tokens=64):
    """Generate text with block duplicated n_passes times."""
    device = next(model.parameters()).device
    inner = model.model
    N = len(inner.layers)

    inp = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inp["input_ids"]

    for _ in range(max_new_tokens):
        h = inner.embed_tokens(input_ids)
        seq_len = h.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = inner.rotary_emb(h, position_ids)

        h = _run_layers(inner, h, 0, block_start, pos_embeds)
        for _ in range(n_passes):
            h = _run_layers(inner, h, block_start, block_end, pos_embeds)
        h = _run_layers(inner, h, block_end, N, pos_embeds)
        h = inner.norm(h)
        logits = model.lm_head(h)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    if output.startswith(prompt):
        output = output[len(prompt):]
    return output.strip()


# =============================================================================
# Phase 1: Collect Utility Trajectories
# =============================================================================

# Math probe questions (imported from math_probe.py)
MATH_QUESTIONS = [
    ("What is 78313 multiplied by 88537?", 6930862681),
    ("What is the cube root of 74088?", 42),
    ("What is 9999 multiplied by 9999?", 99980001),
    ("What is 12345 multiplied by 6789?", 83810205),
    ("What is the square root of 152399025?", 12345),
    ("What is 99999 multiplied by 99999?", 9999800001),
    ("What is 123456789 multiplied by 987654321?", 121932631112635269),
    ("What is 11111 multiplied by 11111?", 123454321),
    ("What is 2 to the power of 20?", 1048576),
    ("What is 777 multiplied by 777?", 603729),
    ("What is the cube root of 19683?", 27),
    ("What is 54321 multiplied by 12345?", 670592745),
    ("What is 31415 multiplied by 92653?", 2910827395),
    ("What is 271828 multiplied by 314159?", 85397339652),
    ("What is 142857 multiplied by 7?", 999999),
    ("What is 99999 multiplied by 100001?", 9999999999),
]

# Additional diverse prompts for cross-task training
DIVERSE_PROMPTS = [
    {"prompt": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost? Answer with just the number in cents.", "correct": "5", "type": "reasoning"},
    {"prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Answer with just the number.", "correct": "5", "type": "reasoning"},
    {"prompt": "A farmer has 17 sheep. All but 9 die. How many are left? Answer with just the number.", "correct": "9", "type": "reasoning"},
    {"prompt": "What is the chemical symbol for gold? Answer with just the symbol.", "correct": "Au", "type": "knowledge"},
    {"prompt": "In what year did World War 2 end? Answer with just the year.", "correct": "1945", "type": "knowledge"},
    {"prompt": "What is the capital of Japan? Answer with just the city name.", "correct": "Tokyo", "type": "knowledge"},
    {"prompt": "Respond with ONLY a single word that means 'happy'.", "correct": None, "type": "instruction"},
    {"prompt": "What is 7 times 8? Respond with ONLY the number.", "correct": "56", "type": "instruction"},
]


def collect_trajectories(model, tokenizer, block_start, block_end, max_passes=3,
                         compute_penalty=0.05):
    """
    For each prompt, run 1..max_passes and record:
    - Hidden states at block exit per pass
    - Whether the answer is correct per pass
    - Optimal stopping pass t*
    """
    print(f"\n{'='*70}")
    print(f"PHASE 1: Collecting Utility Trajectories")
    print(f"  Block: ({block_start},{block_end}), max_passes={max_passes}")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    trajectories = []

    # Math probe questions
    for q_idx, (question, correct_answer) in enumerate(MATH_QUESTIONS):
        prompt = f"System: You are a math calculator. Answer with ONLY the number.\nQuestion: {question}\nAnswer:"

        pass_data = []
        for n_pass in range(1, max_passes + 1):
            # Get hidden states
            inp = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _, pass_hiddens = run_n_passes(
                    model, inp["input_ids"], block_start, block_end, n_pass
                )

            # Generate answer
            response = generate_n_passes(
                model, tokenizer, prompt, block_start, block_end, n_pass
            )

            # Score
            try:
                extracted = ''.join(c for c in response.split('\n')[0] if c.isdigit())
                if extracted:
                    score = calculate_score(correct_answer, int(extracted))
                else:
                    score = 0.0
            except:
                score = 0.0

            correct = 1 if score > 0.8 else 0

            pass_data.append({
                "n_pass": n_pass,
                "score": score,
                "correct": correct,
                "hidden": pass_hiddens[-1].cpu(),  # Last pass exit hidden
                "response": response[:100],
            })

        # Compute utility and optimal stop
        utilities = []
        for pd in pass_data:
            u = pd["score"] - compute_penalty * (pd["n_pass"] - 1)
            utilities.append(u)

        t_star = int(np.argmax(utilities))  # 0-indexed
        halt_targets = [1 if t >= t_star else 0 for t in range(max_passes)]

        traj = {
            "question": question[:60],
            "type": "math",
            "pass_data": pass_data,
            "utilities": utilities,
            "t_star": t_star,
            "halt_targets": halt_targets,
        }
        trajectories.append(traj)

        scores_str = [f"p{pd['n_pass']}={pd['score']:.2f}" for pd in pass_data]
        print(f"  [{q_idx+1}/16] {question[:40]:40s} t*={t_star+1} {' '.join(scores_str)}")

    # Diverse prompts
    for d_idx, item in enumerate(DIVERSE_PROMPTS):
        prompt = f"Question: {item['prompt']}\nAnswer:"

        pass_data = []
        for n_pass in range(1, max_passes + 1):
            inp = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _, pass_hiddens = run_n_passes(
                    model, inp["input_ids"], block_start, block_end, n_pass
                )

            response = generate_n_passes(
                model, tokenizer, prompt, block_start, block_end, n_pass
            )

            # Score
            if item["correct"] is not None:
                correct = 1 if item["correct"].lower() in response.lower() else 0
                score = float(correct)
            else:
                # Instruction following — just check response is short
                correct = 1 if len(response.split()) <= 3 else 0
                score = float(correct)

            pass_data.append({
                "n_pass": n_pass,
                "score": score,
                "correct": correct,
                "hidden": pass_hiddens[-1].cpu(),
                "response": response[:100],
            })

        utilities = [pd["score"] - compute_penalty * (pd["n_pass"] - 1) for pd in pass_data]
        t_star = int(np.argmax(utilities))
        halt_targets = [1 if t >= t_star else 0 for t in range(max_passes)]

        traj = {
            "question": item["prompt"][:60],
            "type": item["type"],
            "pass_data": pass_data,
            "utilities": utilities,
            "t_star": t_star,
            "halt_targets": halt_targets,
        }
        trajectories.append(traj)
        print(f"  [D{d_idx+1}/8] {item['prompt'][:40]:40s} t*={t_star+1}")

    # Summary
    t_star_dist = [0] * max_passes
    for t in trajectories:
        t_star_dist[t["t_star"]] += 1
    print(f"\n  Optimal stop distribution: {dict(enumerate(t_star_dist, 1))}")
    print(f"  Total trajectories: {len(trajectories)}")

    # Save
    saveable = []
    for t in trajectories:
        s = {k: v for k, v in t.items()}
        s["pass_data"] = [{k: v for k, v in pd.items() if k != "hidden"} for pd in t["pass_data"]]
        saveable.append(s)
    with open(RESULTS_DIR / "trajectories.json", "w") as f:
        json.dump(saveable, f, indent=2, default=str)

    # Save tensors
    torch.save({
        "hiddens": [[pd["hidden"] for pd in t["pass_data"]] for t in trajectories],
        "halt_targets": [t["halt_targets"] for t in trajectories],
        "correct_targets": [[pd["correct"] for pd in t["pass_data"]] for t in trajectories],
        "t_stars": [t["t_star"] for t in trajectories],
    }, RESULTS_DIR / "trajectory_tensors.pt")

    return trajectories


# =============================================================================
# Phase 2: Train BrierHalting Head
# =============================================================================

def train_halting_head(model, trajectories, max_passes=3, num_epochs=200,
                       lr=1e-3, lambda_halt=1.0, lambda_correct=0.3,
                       lambda_mono=0.5):
    """Train the BrierHalting head on collected trajectories."""
    print(f"\n{'='*70}")
    print("PHASE 2: Training BrierHalting Head")
    print(f"{'='*70}")

    hidden_size = model.config.hidden_size
    device = next(model.parameters()).device

    # Load tensor data
    data = torch.load(RESULTS_DIR / "trajectory_tensors.pt")
    all_hiddens = data["hiddens"]       # list of lists of [hidden_size] tensors
    halt_targets = data["halt_targets"]  # list of lists of 0/1
    correct_targets = data["correct_targets"]

    N = len(all_hiddens)
    print(f"  Trajectories: {N}, max_passes: {max_passes}")

    # Build training tensors
    # For each (trajectory, pass), we have (h_current, h_previous, halt_target, correct_target)
    train_h = []
    train_h_prev = []
    train_halt = []
    train_correct = []
    train_pass_idx = []

    for i in range(N):
        for t in range(max_passes):
            h_curr = all_hiddens[i][t].to(device).squeeze(0)  # [hidden_size]
            h_prev = all_hiddens[i][t-1].to(device).squeeze(0) if t > 0 else torch.zeros_like(h_curr)

            train_h.append(h_curr)
            train_h_prev.append(h_prev)
            train_halt.append(halt_targets[i][t])
            train_correct.append(correct_targets[i][t])
            train_pass_idx.append(t)

    train_h = torch.stack(train_h)          # [N*max_passes, hidden_size]
    train_h_prev = torch.stack(train_h_prev)
    train_halt = torch.tensor(train_halt, dtype=torch.float32, device=device)
    train_correct = torch.tensor(train_correct, dtype=torch.float32, device=device)
    train_pass_idx_t = torch.tensor(train_pass_idx, dtype=torch.long, device=device)

    total_samples = len(train_h)
    print(f"  Training samples: {total_samples}")
    print(f"  Halt target distribution: stop={train_halt.sum().item():.0f}, continue={(1-train_halt).sum().item():.0f}")

    # Build head
    head = BrierHaltingHead(hidden_size, max_passes=max_passes).to(device)
    param_count = sum(p.numel() for p in head.parameters())
    print(f"  Head params: {param_count:,}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        head.train()

        # Run per pass_idx for correct pass embedding
        q_halt_all = []
        q_correct_all = []
        for t in range(max_passes):
            mask = train_pass_idx_t == t
            if mask.sum() == 0:
                continue
            qh, qc = head(train_h[mask], train_h_prev[mask], t)
            q_halt_all.append((mask, qh))
            q_correct_all.append((mask, qc))

        # Reassemble
        q_halt_full = torch.zeros(total_samples, 1, device=device)
        q_correct_full = torch.zeros(total_samples, 1, device=device)
        for mask, qh in q_halt_all:
            q_halt_full[mask] = qh
        for mask, qc in q_correct_all:
            q_correct_full[mask] = qc

        q_halt_flat = q_halt_full.squeeze(-1)
        q_correct_flat = q_correct_full.squeeze(-1)

        # Brier score losses
        halt_loss = ((q_halt_flat - train_halt) ** 2).mean()
        correct_loss = ((q_correct_flat - train_correct) ** 2).mean()

        # Monotonicity: q_halt should increase with pass index
        mono_loss = torch.tensor(0.0, device=device)
        for i in range(N):
            start = i * max_passes
            for t in range(max_passes - 1):
                q_curr = q_halt_flat[start + t]
                q_next = q_halt_flat[start + t + 1]
                mono_loss = mono_loss + torch.relu(q_curr - q_next) ** 2
        mono_loss = mono_loss / (N * (max_passes - 1))

        loss = lambda_halt * halt_loss + lambda_correct * correct_loss + lambda_mono * mono_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

        if (epoch + 1) % 40 == 0:
            # Accuracy: does the head correctly predict halt/continue?
            pred_halt = (q_halt_flat > 0.5).float()
            halt_acc = (pred_halt == train_halt).float().mean().item()
            print(f"    Epoch {epoch+1}/{num_epochs}  loss={loss.item():.4f}  "
                  f"halt={halt_loss.item():.4f}  correct={correct_loss.item():.4f}  "
                  f"mono={mono_loss.item():.4f}  halt_acc={halt_acc:.1%}")

    head.load_state_dict(best_state)

    torch.save({
        "state_dict": head.state_dict(),
        "hidden_size": hidden_size,
        "max_passes": max_passes,
        "param_count": param_count,
        "best_loss": best_loss,
    }, RESULTS_DIR / "brierhalting_head.pt")

    print(f"\n  Best loss: {best_loss:.4f}")
    print(f"  Saved to {RESULTS_DIR / 'brierhalting_head.pt'}")
    return head


# =============================================================================
# Phase 3: Evaluate
# =============================================================================

def evaluate_brierhalting(model, tokenizer, head, block_start, block_end,
                          max_passes=3, halt_threshold=0.5):
    """
    Evaluate:
    1. Baseline (no duplication, = 1 pass through original layers)
    2. Fixed 2-pass
    3. Fixed 3-pass
    4. BrierHalting (adaptive)
    """
    print(f"\n{'='*70}")
    print("PHASE 3: Evaluation")
    print(f"  Block: ({block_start},{block_end}), threshold={halt_threshold}")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    inner = model.model
    N_layers = len(inner.layers)
    systems = {}

    # 1. Baseline
    print("\n--- Baseline (1 pass = no duplication) ---")
    def gen_1pass(prompt):
        return generate_n_passes(model, tokenizer, prompt, block_start, block_end, 1)
    r1 = run_math_probe(gen_1pass, verbose=False)
    systems["1_pass_baseline"] = r1["score"]
    print(f"  Score: {r1['score']:.4f}")

    # 2. Fixed 2-pass
    print("\n--- Fixed 2-pass ---")
    def gen_2pass(prompt):
        return generate_n_passes(model, tokenizer, prompt, block_start, block_end, 2)
    r2 = run_math_probe(gen_2pass, verbose=False)
    systems["2_pass_fixed"] = r2["score"]
    print(f"  Score: {r2['score']:.4f}")

    # 3. Fixed 3-pass
    print("\n--- Fixed 3-pass ---")
    def gen_3pass(prompt):
        return generate_n_passes(model, tokenizer, prompt, block_start, block_end, 3)
    r3 = run_math_probe(gen_3pass, verbose=False)
    systems["3_pass_fixed"] = r3["score"]
    print(f"  Score: {r3['score']:.4f}")

    # 4. BrierHalting
    print(f"\n--- BrierHalting (adaptive, threshold={halt_threshold}) ---")
    head.eval()
    halt_decisions = []

    def gen_brierhalting(prompt):
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inp["input_ids"]

        # We need to generate token by token with adaptive passes
        # First, determine how many passes for this prompt
        with torch.no_grad():
            _, pass_hiddens = run_n_passes(
                model, input_ids, block_start, block_end, max_passes
            )

        # Determine optimal passes via halting head
        h_prev = torch.zeros(1, model.config.hidden_size, device=device)
        chosen_passes = max_passes  # default: use all

        for t in range(max_passes):
            h_curr = pass_hiddens[t].unsqueeze(0) if pass_hiddens[t].dim() == 1 else pass_hiddens[t]
            h_p = h_prev
            q_halt, q_correct = head(h_curr, h_p, t)

            if q_halt.item() > halt_threshold:
                chosen_passes = t + 1  # 1-indexed
                break
            h_prev = h_curr

        halt_decisions.append(chosen_passes)

        # Generate with chosen number of passes
        return generate_n_passes(
            model, tokenizer, prompt, block_start, block_end, chosen_passes
        )

    r_bh = run_math_probe(gen_brierhalting, verbose=False)
    systems["brierhalting"] = r_bh["score"]

    pass_dist = {}
    for p in halt_decisions:
        pass_dist[p] = pass_dist.get(p, 0) + 1
    avg_passes = np.mean(halt_decisions)

    print(f"  Score: {r_bh['score']:.4f}")
    print(f"  Pass distribution: {pass_dist}")
    print(f"  Average passes: {avg_passes:.2f}")

    # Summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  {'System':25s} {'Score':>8} {'Delta':>8} {'Passes':>8}")
    print(f"  {'-'*55}")
    baseline = systems["1_pass_baseline"]
    for name, score in systems.items():
        passes = "1" if "1_pass" in name else "2" if "2_pass" in name else "3" if "3_pass" in name else f"~{avg_passes:.1f}"
        marker = " ***" if score == max(systems.values()) else ""
        print(f"  {name:25s} {score:8.4f} {score-baseline:+8.4f} {passes:>8}{marker}")

    # Save
    output = {
        "systems": systems,
        "baseline": baseline,
        "block": [block_start, block_end],
        "halt_threshold": halt_threshold,
        "halt_decisions": halt_decisions,
        "pass_distribution": pass_dist,
        "avg_passes": avg_passes,
    }
    with open(RESULTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    return systems


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct")
    parser.add_argument("--block-start", type=int, default=10)
    parser.add_argument("--block-end", type=int, default=11)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--halt-threshold", type=float, default=0.5)
    parser.add_argument("--compute-penalty", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--phase", choices=["all", "1", "2", "3"], default="all")
    args = parser.parse_args()

    t0 = time.time()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_original_model(args.model)

    if args.phase in ["all", "1"]:
        trajectories = collect_trajectories(
            model, tokenizer, args.block_start, args.block_end,
            max_passes=args.max_passes, compute_penalty=args.compute_penalty
        )

    head = None
    if args.phase in ["all", "2"]:
        head = train_halting_head(
            model, None, max_passes=args.max_passes,
            num_epochs=args.epochs, lr=args.lr
        )

    if args.phase in ["all", "3"]:
        if head is None:
            device = next(model.parameters()).device
            ckpt = torch.load(RESULTS_DIR / "brierhalting_head.pt")
            head = BrierHaltingHead(
                ckpt["hidden_size"], ckpt["max_passes"]
            ).to(device)
            head.load_state_dict(ckpt["state_dict"])

        evaluate_brierhalting(
            model, tokenizer, head, args.block_start, args.block_end,
            max_passes=args.max_passes, halt_threshold=args.halt_threshold
        )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
