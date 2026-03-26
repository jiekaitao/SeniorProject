"""
Adaptive Iteration Router — Full ESR + DSG Hybrid Implementation

Based on GPT-5.4 Pro's design with corrections from diagnostic findings:
  1. ESR scoring uses LM-head margin gain (not just geometric rho/residual)
  2. "No duplication" is arm 0 (first-class, not ablation)
  3. DSG uses last-token + learned projection (not wide 4x pooling)
  4. Cascaded hybrid: DSG when confident, ESR fallback on top-2

Pipeline:
  Phase 1: Collect ESR teacher labels on diverse prompt bank
  Phase 2: Train DSG router on ESR labels
  Phase 3: Evaluate all systems (baseline, fixed, ESR oracle, DSG, hybrid)
"""

import sys, os, json, time, torch, gc, argparse
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
# Also add flat scripts dir for backward compat
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts'))

from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe

RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/adaptive_router")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Block Specifications
# =============================================================================

BLOCKS_7B = [
    # name, start, end, prior_score (from brain scanner / spectral)
    {"name": "no_dup", "start": -1, "end": -1, "prior": 0.0},
    {"name": "mid_good_10_11", "start": 10, "end": 11, "prior": 0.2571},
    {"name": "mid_wide_14_17", "start": 14, "end": 17, "prior": 0.0},
    {"name": "late_good_18_21", "start": 18, "end": 21, "prior": 0.2349},
]


# =============================================================================
# Prompt Bank — Diverse mixed tasks for ESR label collection
# =============================================================================

PROMPT_BANK = {
    "arithmetic": [
        "What is 78313 multiplied by 88537?",
        "What is 9999 multiplied by 9999?",
        "The cube root of 74088 is approximately",
        "The square root of 152399025 is",
        "What is 123456789 multiplied by 987654321?",
        "What is 31415 divided by 271?",
        "What is 2 to the power of 17?",
        "What is 999999 times 7?",
        "What is 45678 plus 87654?",
        "What is the square of 314?",
    ],
    "reasoning": [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "A farmer has 17 sheep. All but 9 die. How many are left?",
        "If you overtake the person in 2nd place in a race, what place are you in?",
        "How many times can you subtract 5 from 25?",
        "Three siblings have ages that multiply to 36 and add to the house number next door. What are the ages?",
        "You have two ropes that each take exactly 1 hour to burn. How do you measure 45 minutes?",
        "A snail climbs 3 feet up a well each day and slides back 2 feet each night. The well is 30 feet deep. How many days?",
        "If you have a 3-gallon and a 5-gallon jug, how do you measure exactly 4 gallons?",
    ],
    "instruction": [
        "Write exactly three sentences about the moon. Each sentence must start with a different letter.",
        "List five fruits in alphabetical order, separated by semicolons.",
        "Respond with only a single word that means 'happy'.",
        "Write a haiku about programming. Remember: 5-7-5 syllable structure.",
        "Name three countries that start with the letter B, one per line.",
        "Translate 'hello world' to French, Spanish, and German. Format as a numbered list.",
        "Write the alphabet backwards, with a comma between each letter.",
        "Give exactly two reasons why the sky is blue. Number them.",
        "Write a sentence that contains exactly 10 words.",
        "List the planets in order from the sun, comma-separated.",
    ],
    "knowledge": [
        "What is the speed of light in meters per second?",
        "Who wrote the novel '1984'?",
        "What is the chemical formula for sulfuric acid?",
        "In what year did the Berlin Wall fall?",
        "What is the capital of Australia?",
        "What is the atomic number of gold?",
        "Who painted the Sistine Chapel ceiling?",
        "What is the largest organ in the human body?",
        "What is the boiling point of water in Fahrenheit?",
        "Who developed the theory of general relativity?",
    ],
    "coding": [
        "Write a Python function that checks if a number is prime.",
        "Implement binary search in Python.",
        "Write a function to reverse a linked list.",
        "Write Python code to flatten a nested list.",
        "Implement a stack using two queues.",
        "Write a function to check if a string is a valid palindrome.",
        "How do you detect a cycle in a linked list?",
        "Write Python code to find the two numbers in a list that sum to a target.",
        "Implement merge sort in Python.",
        "Write a Python function to compute Fibonacci numbers with memoization.",
    ],
}


# =============================================================================
# Exact Spectral Router (ESR)
# =============================================================================

class ExactSpectralRouter:
    """
    Measures displacement rho + margin gain for each candidate block.
    Uses the corrected V2 scoring from diagnostic findings.
    """

    def __init__(self, model, blocks, anchor_layer=14):
        self.model = model
        self.inner = model.model
        self.blocks = blocks
        self.anchor_layer = anchor_layer
        self.lm_head = model.lm_head
        self.final_norm = model.model.norm
        self.device = next(model.parameters()).device

    def _run_layers(self, h, start, end, pos_embeds):
        """Run layers [start, end) on hidden state h."""
        for i in range(start, end):
            out = self.inner.layers[i](h, position_embeddings=pos_embeds, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out
        return h

    def _margin(self, h):
        """Top-1 minus top-2 logit from LM head."""
        h_normed = self.final_norm(h[:, -1:, :])
        logits = self.lm_head(h_normed)[:, -1, :].float()
        vals = logits.topk(k=2, dim=-1).values
        return (vals[:, 0] - vals[:, 1]).mean().item()

    def _flat_norm(self, t):
        return t.float().reshape(t.shape[0], -1).norm(dim=-1).item()

    @torch.no_grad()
    def score_all_blocks(self, input_ids, attention_mask=None):
        """
        Run a single forward pass caching block boundaries,
        then score each candidate block.

        Returns:
            scores: dict {block_name: score}
            details: dict {block_name: {rho, residual, margin_gain, score}}
            anchor_hidden: hidden state at anchor layer (for DSG training)
            baseline_output: final hidden state without duplication
        """
        N = len(self.inner.layers)
        h = self.inner.embed_tokens(input_ids.to(self.device))
        seq_len = h.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        pos_embeds = self.inner.rotary_emb(h, position_ids)

        # Collect all needed boundary indices
        boundaries = set([0])
        for block in self.blocks:
            if block["start"] >= 0:
                boundaries.add(block["start"])
                boundaries.add(block["end"])
        boundaries.add(self.anchor_layer)
        boundaries.add(N)
        boundaries = sorted(boundaries)

        # Run forward, caching at boundaries
        cache = {}
        cache[0] = h
        for i in range(len(boundaries) - 1):
            start_b, end_b = boundaries[i], boundaries[i + 1]
            h = self._run_layers(h, start_b, end_b, pos_embeds)
            cache[end_b] = h.detach().clone()

        # Baseline margin (no duplication)
        baseline_h = cache[N]
        baseline_margin = self._margin(baseline_h)

        # Extract anchor hidden state for DSG
        anchor_hidden = cache.get(self.anchor_layer)
        if anchor_hidden is None:
            # Anchor is between boundaries, compute it
            prev = max(b for b in boundaries if b <= self.anchor_layer)
            anchor_hidden = self._run_layers(cache[prev], prev, self.anchor_layer, pos_embeds)

        scores = {}
        details = {}

        for block in self.blocks:
            name = block["name"]

            if block["start"] < 0:
                # No-dup arm: score = 0 (baseline reference)
                scores[name] = 0.0
                details[name] = {
                    "rho": 1.0, "residual": 0.0,
                    "margin_gain": 0.0, "score": 0.0
                }
                continue

            s, e = block["start"], block["end"]
            h_before = cache[s]

            # First pass
            h1 = self._run_layers(h_before, s, e, pos_embeds)

            # Second pass (duplication)
            h2 = self._run_layers(h1, s, e, pos_embeds)

            # Displacement rho
            d1 = self._flat_norm(h1 - h_before)
            d2 = self._flat_norm(h2 - h1)
            rho = d2 / (d1 + 1e-6)
            residual = np.log1p(d2)

            # Run suffix to get duplicated output margin
            h_suffix = self._run_layers(h2, e, N, pos_embeds)
            dup_margin = self._margin(h_suffix)
            margin_gain = dup_margin - baseline_margin

            # V2 scoring: margin-dominant
            score = (0.50 * margin_gain +
                     0.30 * (1.0 - rho) +
                     0.20 * min(residual, 6.0) +
                     0.05 * block["prior"])

            scores[name] = score
            details[name] = {
                "rho": rho, "residual": residual,
                "margin_gain": margin_gain, "score": score
            }

        return scores, details, anchor_hidden, baseline_h

    def select_best(self, input_ids, attention_mask=None):
        """Return the best block name and its score."""
        scores, details, anchor, _ = self.score_all_blocks(input_ids, attention_mask)
        best = max(scores, key=scores.get)
        return best, scores[best], details, anchor


# =============================================================================
# Distilled Spectral Gate (DSG)
# =============================================================================

class DistilledSpectralGate(nn.Module):
    """
    Learned router: predicts best block from anchor hidden state.
    Uses last-token + learned projection (not wide pooling).
    """

    def __init__(self, hidden_dim, num_blocks, proj_dim=256):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.act = nn.GELU()
        self.head = nn.Linear(proj_dim, num_blocks)

    def forward(self, anchor_hidden, prior_scores=None):
        """
        Args:
            anchor_hidden: [B, seq_len, hidden_dim] from anchor layer
            prior_scores: [num_blocks] offline spectral priors
        Returns:
            logits: [B, num_blocks]
            probs: [B, num_blocks]
        """
        # Last token only
        h = anchor_hidden[:, -1, :]  # [B, hidden_dim]
        h = self.act(self.proj(h))   # [B, proj_dim]
        logits = self.head(h)        # [B, num_blocks]

        if prior_scores is not None:
            logits = logits + prior_scores.unsqueeze(0)

        probs = logits.softmax(dim=-1)
        return logits, probs


# =============================================================================
# Full Forward with Routing
# =============================================================================

def forward_with_block(model, input_ids, block, pos_embeds=None):
    """
    Run model with a specific block duplicated.
    Returns generated text via generate_no_cache.
    """
    if block["start"] < 0:
        # No duplication — normal forward
        return None  # Signal to use normal generation

    inner = model.model
    N = len(inner.layers)
    s, e = block["start"], block["end"]
    device = next(model.parameters()).device

    h = inner.embed_tokens(input_ids.to(device))
    seq_len = h.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    pos_embeds = inner.rotary_emb(h, position_ids)

    # Run prefix
    h = _run_layer_range(inner, h, 0, s, pos_embeds)
    # First pass
    h = _run_layer_range(inner, h, s, e, pos_embeds)
    # Second pass (duplication)
    h = _run_layer_range(inner, h, s, e, pos_embeds)
    # Suffix
    h = _run_layer_range(inner, h, e, N, pos_embeds)

    h = inner.norm(h)
    logits = model.lm_head(h)
    return logits


def _run_layer_range(inner, h, start, end, pos_embeds):
    for i in range(start, end):
        out = inner.layers[i](h, position_embeddings=pos_embeds, use_cache=False)
        h = out[0] if isinstance(out, tuple) else out
    return h


def generate_with_block(model, tokenizer, prompt, block, max_new_tokens=64):
    """Generate text with a specific block duplicated, token by token."""
    device = next(model.parameters()).device
    inner = model.model

    if block["start"] < 0:
        # No duplication
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=max_new_tokens)

    s, e = block["start"], block["end"]
    N = len(inner.layers)

    inp = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inp["input_ids"]

    for _ in range(max_new_tokens):
        h = inner.embed_tokens(input_ids)
        seq_len = h.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = inner.rotary_emb(h, position_ids)

        h = _run_layer_range(inner, h, 0, s, pos_embeds)
        h = _run_layer_range(inner, h, s, e, pos_embeds)
        h = _run_layer_range(inner, h, s, e, pos_embeds)  # duplication
        h = _run_layer_range(inner, h, e, N, pos_embeds)
        h = inner.norm(h)
        logits = model.lm_head(h)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    # Remove the prompt from output
    if output.startswith(prompt):
        output = output[len(prompt):]
    return output.strip()


# =============================================================================
# Phase 1: Collect ESR Teacher Labels
# =============================================================================

def phase1_collect_labels(model, tokenizer, blocks, anchor_layer=14):
    """Collect ESR scores for all prompts in the bank."""
    print(f"\n{'='*70}")
    print("PHASE 1: Collecting ESR Teacher Labels")
    print(f"{'='*70}")

    esr = ExactSpectralRouter(model, blocks, anchor_layer=anchor_layer)
    block_names = [b["name"] for b in blocks]

    all_labels = []
    for task_name, prompts in PROMPT_BANK.items():
        print(f"\n  Task: {task_name}")
        for i, prompt in enumerate(prompts):
            inp = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=64)

            scores, details, anchor_h, _ = esr.score_all_blocks(inp["input_ids"])
            best = max(scores, key=scores.get)
            best_idx = block_names.index(best)

            # Normalize scores to probability distribution
            score_vec = torch.tensor([scores[n] for n in block_names])
            score_probs = torch.softmax(score_vec * 2.0, dim=-1)  # temperature=0.5

            label = {
                "task": task_name,
                "prompt": prompt[:80],
                "scores": {n: scores[n] for n in block_names},
                "details": details,
                "best_block": best,
                "best_idx": best_idx,
                "score_probs": score_probs.tolist(),
                "anchor_hidden": anchor_h[0, -1, :].cpu(),  # Last token, squeeze batch dim → [hidden_dim]
            }
            all_labels.append(label)

            if (i + 1) % 5 == 0:
                print(f"    [{i+1}/{len(prompts)}] Best: {best} "
                      f"(margin: {details[best].get('margin_gain', 0):.2f})")

    # Summary
    print(f"\n  Total labels: {len(all_labels)}")
    winners = defaultdict(int)
    for l in all_labels:
        winners[l["best_block"]] += 1
    print(f"  Winner distribution: {dict(winners)}")

    # Save labels (without anchor_hidden tensors for JSON)
    saveable = []
    for l in all_labels:
        s = {k: v for k, v in l.items() if k != "anchor_hidden"}
        saveable.append(s)
    with open(RESULTS_DIR / "esr_labels.json", "w") as f:
        json.dump(saveable, f, indent=2, default=str)

    # Save tensor data separately
    torch.save(
        {"anchors": [l["anchor_hidden"] for l in all_labels],
         "best_idxs": [l["best_idx"] for l in all_labels],
         "score_probs": [l["score_probs"] for l in all_labels]},
        RESULTS_DIR / "esr_training_data.pt"
    )

    print(f"  Saved to {RESULTS_DIR}")
    return all_labels


# =============================================================================
# Phase 2: Train DSG Router
# =============================================================================

def phase2_train_dsg(model, blocks, num_epochs=100, lr=1e-3):
    """Train DSG router on ESR teacher labels."""
    print(f"\n{'='*70}")
    print("PHASE 2: Training Distilled Spectral Gate")
    print(f"{'='*70}")

    hidden_dim = model.config.hidden_size
    num_blocks = len(blocks)
    device = next(model.parameters()).device

    # Load training data
    data = torch.load(RESULTS_DIR / "esr_training_data.pt")
    anchors = torch.stack(data["anchors"]).to(device)       # [N, hidden_dim]
    best_idxs = torch.tensor(data["best_idxs"], dtype=torch.long).to(device)  # [N]
    score_probs = torch.tensor(data["score_probs"], dtype=torch.float32).to(device)  # [N, num_blocks]

    N = len(anchors)
    print(f"  Training samples: {N}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num blocks (incl no-dup): {num_blocks}")

    # Build DSG (bfloat16 to match model hidden states)
    dsg = DistilledSpectralGate(hidden_dim, num_blocks, proj_dim=256).to(device).to(torch.bfloat16)

    # Prior scores
    prior = torch.tensor([b["prior"] for b in blocks], device=device)

    optimizer = torch.optim.AdamW(dsg.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training
    best_acc = 0
    best_state = None

    for epoch in range(num_epochs):
        dsg.train()

        # Add seq_len dimension for DSG: [N, hidden_dim] → [N, 1, hidden_dim]
        anchor_input = anchors.unsqueeze(1)  # [N, 1, hidden_dim]
        logits, probs = dsg(anchor_input, prior_scores=prior)
        # logits: [N, num_blocks], probs: [N, num_blocks]

        # Cast to float32 for loss computation
        logits_f = logits.float()
        probs_f = probs.float()

        # Loss: CE with hard labels + KL with soft labels
        ce_loss = nn.functional.cross_entropy(logits_f, best_idxs)
        kl_loss = nn.functional.kl_div(
            probs_f.log().clamp(min=-100),
            score_probs,
            reduction='batchmean'
        )
        # Entropy bonus for exploration
        ent_loss = -(probs_f * probs_f.log().clamp(min=-100)).sum(dim=-1).mean()

        loss = 0.5 * ce_loss + 0.5 * kl_loss - 0.01 * ent_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(dsg.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Accuracy
        preds = logits_f.argmax(dim=-1)
        acc = (preds == best_idxs).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in dsg.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            conf = probs.max(dim=-1).values.mean().item()
            print(f"    Epoch {epoch+1}/{num_epochs}  loss={loss.item():.4f}  "
                  f"acc={acc:.1%}  conf={conf:.2f}  "
                  f"ce={ce_loss.item():.4f}  kl={kl_loss.item():.4f}")

    # Load best
    dsg.load_state_dict(best_state)
    print(f"\n  Best accuracy: {best_acc:.1%}")

    # Save
    torch.save({
        "state_dict": dsg.state_dict(),
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "proj_dim": 256,
        "best_acc": best_acc,
    }, RESULTS_DIR / "dsg_router.pt")

    return dsg


# =============================================================================
# Phase 3: Evaluate All Systems
# =============================================================================

def phase3_evaluate(model, tokenizer, blocks, dsg, anchor_layer=14):
    """
    Evaluate:
    1. Baseline (no duplication)
    2. Fixed best block (10,11)
    3. Fixed second block (18,21)
    4. ESR oracle (picks best per prompt)
    5. DSG learned router
    6. Hybrid (DSG + ESR fallback)
    """
    print(f"\n{'='*70}")
    print("PHASE 3: Evaluation on Math Probe")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    block_names = [b["name"] for b in blocks]
    esr = ExactSpectralRouter(model, blocks, anchor_layer=anchor_layer)
    prior = torch.tensor([b["prior"] for b in blocks], device=device)

    # System definitions
    systems = {}

    # 1. Baseline
    print("\n--- System 1: Baseline (no duplication) ---")
    def gen_baseline(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    baseline_result = run_math_probe(gen_baseline, verbose=False)
    systems["baseline"] = baseline_result["score"]
    print(f"  Score: {baseline_result['score']:.4f}")

    # 2. Fixed (10,11)
    print("\n--- System 2: Fixed (10,11) ---")
    block_10_11 = blocks[1]  # mid_good_10_11
    def gen_fixed_10_11(prompt):
        return generate_with_block(model, tokenizer, prompt, block_10_11)
    fixed_10_11_result = run_math_probe(gen_fixed_10_11, verbose=False)
    systems["fixed_10_11"] = fixed_10_11_result["score"]
    print(f"  Score: {fixed_10_11_result['score']:.4f}")

    # 3. Fixed (18,21)
    print("\n--- System 3: Fixed (18,21) ---")
    block_18_21 = blocks[3]  # late_good_18_21
    def gen_fixed_18_21(prompt):
        return generate_with_block(model, tokenizer, prompt, block_18_21)
    fixed_18_21_result = run_math_probe(gen_fixed_18_21, verbose=False)
    systems["fixed_18_21"] = fixed_18_21_result["score"]
    print(f"  Score: {fixed_18_21_result['score']:.4f}")

    # 4. ESR Oracle
    print("\n--- System 4: ESR Oracle (per-prompt best) ---")
    esr_decisions = {}

    def gen_esr_oracle(prompt):
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        best_name, score, details, _ = esr.select_best(inp["input_ids"])
        esr_decisions[prompt[:50]] = best_name
        block = next(b for b in blocks if b["name"] == best_name)
        return generate_with_block(model, tokenizer, prompt, block)

    esr_result = run_math_probe(gen_esr_oracle, verbose=False)
    systems["esr_oracle"] = esr_result["score"]
    print(f"  Score: {esr_result['score']:.4f}")
    print(f"  Decisions: {dict(defaultdict(int, {v: sum(1 for x in esr_decisions.values() if x == v) for v in set(esr_decisions.values())}))}")

    # 5. DSG Learned Router
    print("\n--- System 5: DSG Learned Router ---")
    dsg.eval()
    dsg_decisions = {}

    def gen_dsg(prompt):
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inp["input_ids"].to(device)

        # Get anchor hidden state
        h = model.model.embed_tokens(input_ids)
        seq_len = h.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = model.model.rotary_emb(h, position_ids)
        h = _run_layer_range(model.model, h, 0, anchor_layer, pos_embeds)

        with torch.no_grad():
            logits, probs = dsg(h, prior_scores=prior)
        choice = logits.argmax(dim=-1).item()
        confidence = probs.max(dim=-1).values.item()

        chosen_block = blocks[choice]
        dsg_decisions[prompt[:50]] = (chosen_block["name"], confidence)
        return generate_with_block(model, tokenizer, prompt, chosen_block)

    dsg_result = run_math_probe(gen_dsg, verbose=False)
    systems["dsg_router"] = dsg_result["score"]
    print(f"  Score: {dsg_result['score']:.4f}")
    dsg_names = [v[0] for v in dsg_decisions.values()]
    print(f"  Decisions: {dict(defaultdict(int, {v: sum(1 for x in dsg_names if x == v) for v in set(dsg_names)}))}")
    avg_conf = np.mean([v[1] for v in dsg_decisions.values()])
    print(f"  Avg confidence: {avg_conf:.2f}")

    # 6. Hybrid (DSG + ESR fallback)
    print("\n--- System 6: Hybrid (DSG + ESR top-2 fallback) ---")
    CONFIDENCE_THRESHOLD = 0.60
    hybrid_decisions = {}
    fallback_count = 0
    total_count = 0

    def gen_hybrid(prompt):
        nonlocal fallback_count, total_count
        total_count += 1

        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inp["input_ids"].to(device)

        # Get anchor hidden state
        h = model.model.embed_tokens(input_ids)
        seq_len = h.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = model.model.rotary_emb(h, position_ids)
        h = _run_layer_range(model.model, h, 0, anchor_layer, pos_embeds)

        with torch.no_grad():
            logits, probs = dsg(h, prior_scores=prior)
        top_conf = probs.max(dim=-1).values.item()
        top_choice = logits.argmax(dim=-1).item()

        if top_conf >= CONFIDENCE_THRESHOLD:
            # DSG is confident — use its choice
            chosen = blocks[top_choice]
            hybrid_decisions[prompt[:50]] = (chosen["name"], "dsg", top_conf)
        else:
            # Fallback to ESR on top-2
            fallback_count += 1
            top2 = logits.topk(k=min(2, len(blocks)), dim=-1).indices[0].tolist()

            best_score = float('-inf')
            chosen = blocks[0]  # default no-dup

            scores, details, _, _ = esr.score_all_blocks(input_ids)
            for idx in top2:
                name = block_names[idx]
                if scores[name] > best_score:
                    best_score = scores[name]
                    chosen = blocks[idx]

            hybrid_decisions[prompt[:50]] = (chosen["name"], "esr_fallback", top_conf)

        return generate_with_block(model, tokenizer, prompt, chosen)

    hybrid_result = run_math_probe(gen_hybrid, verbose=False)
    systems["hybrid"] = hybrid_result["score"]
    fallback_rate = fallback_count / max(total_count, 1)
    print(f"  Score: {hybrid_result['score']:.4f}")
    print(f"  Fallback rate: {fallback_rate:.1%} ({fallback_count}/{total_count})")
    hybrid_names = [v[0] for v in hybrid_decisions.values()]
    print(f"  Decisions: {dict(defaultdict(int, {v: sum(1 for x in hybrid_names if x == v) for v in set(hybrid_names)}))}")

    # Final comparison
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  {'System':25s} {'Score':>8} {'Delta':>8}")
    print(f"  {'-'*45}")
    baseline_score = systems["baseline"]
    for name, score in systems.items():
        delta = score - baseline_score
        marker = " ***" if score == max(systems.values()) else ""
        print(f"  {name:25s} {score:8.4f} {delta:+8.4f}{marker}")

    # Save
    output = {
        "systems": systems,
        "baseline": baseline_score,
        "esr_decisions": {k: v for k, v in esr_decisions.items()},
        "dsg_decisions": {k: list(v) for k, v in dsg_decisions.items()},
        "hybrid_decisions": {k: list(v) for k, v in hybrid_decisions.items()},
        "fallback_rate": fallback_rate,
        "dsg_avg_confidence": avg_conf,
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
    parser.add_argument("--anchor-layer", type=int, default=14)
    parser.add_argument("--dsg-epochs", type=int, default=100)
    parser.add_argument("--dsg-lr", type=float, default=1e-3)
    parser.add_argument("--phase", choices=["all", "1", "2", "3"], default="all")
    args = parser.parse_args()

    blocks = BLOCKS_7B

    t0 = time.time()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_original_model(args.model)

    if args.phase in ["all", "1"]:
        phase1_collect_labels(model, tokenizer, blocks, anchor_layer=args.anchor_layer)

    dsg = None
    if args.phase in ["all", "2"]:
        dsg = phase2_train_dsg(model, blocks, num_epochs=args.dsg_epochs, lr=args.dsg_lr)

    if args.phase in ["all", "3"]:
        if dsg is None:
            # Load saved DSG
            device = next(model.parameters()).device
            ckpt = torch.load(RESULTS_DIR / "dsg_router.pt")
            dsg = DistilledSpectralGate(
                ckpt["hidden_dim"], ckpt["num_blocks"], ckpt["proj_dim"]
            ).to(device).to(torch.bfloat16)
            dsg.load_state_dict(ckpt["state_dict"])

        phase3_evaluate(model, tokenizer, blocks, dsg, anchor_layer=args.anchor_layer)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
