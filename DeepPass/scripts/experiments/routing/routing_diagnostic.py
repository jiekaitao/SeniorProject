"""
Routing Diagnostic — Does the optimal block vary per-input or per-task?

Before building any learned router, answer the critical feasibility question:
is there enough per-input signal to justify geometric routing, or is a simple
task classifier sufficient?

This script:
1. Defines a mixed prompt bank across task families
2. For each prompt, computes ESR scores for K candidate blocks on 7B
3. Measures:
   - Per-task dominance: does one block always win within a task family?
   - Conditional entropy H(B* | T): how much block uncertainty remains after knowing the task?
   - Within-task argmax stability: fraction of prompts where the winner changes
   - Score vector variance: is the full score distribution task-determined?
4. Produces a clear verdict: task-classifier vs geometric router

Run on 7B (cheap, ~30 min). If within-task variation is high, the router is justified.
If block choice is >85% determined by task family, just build a task classifier.
"""

import sys, os, json, time, torch, gc
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_duplicator import load_original_model, generate_no_cache
import copy
import torch.nn as nn

RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/routing_diagnostic")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Candidate blocks for 7B (28 layers) — diverse regions
# Good: (10,11), (18,21) from brain scanner
# Bad: (4,9) from brain scanner
# Mid: (14,17) moderate region
CANDIDATE_BLOCKS_7B = [
    {"name": "early_bad", "start": 4, "end": 9},
    {"name": "mid_good", "start": 10, "end": 11},
    {"name": "mid_wide", "start": 14, "end": 17},
    {"name": "late_good", "start": 18, "end": 21},
]

# Task-family prompt bank — 8 prompts per family, 6 families = 48 prompts
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
    ],
    "reasoning": [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "Three friends split a $30 bill. They each pay $10. The waiter returns $5. They each take $1 back and tip $2. Where did the extra dollar go?",
        "You have 8 balls, one is heavier. You have a balance scale. What's the minimum weighings to find the heavy ball?",
        "A farmer has 17 sheep. All but 9 die. How many are left?",
        "If you overtake the person in 2nd place in a race, what place are you in?",
        "How many times can you subtract 5 from 25?",
    ],
    "instruction_following": [
        "Write exactly three sentences about the moon. Each sentence must start with a different letter.",
        "List five fruits in alphabetical order, separated by semicolons.",
        "Respond with only a single word that means 'happy'.",
        "Write a haiku about programming. Remember: 5-7-5 syllable structure.",
        "Give me a paragraph with exactly 50 words about climate change.",
        "Name three countries that start with the letter B, one per line.",
        "Translate 'hello world' to French, Spanish, and German. Format as a numbered list.",
        "Write the alphabet backwards, with a comma between each letter.",
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
    ],
    "coding": [
        "Write a Python function that checks if a number is prime.",
        "Implement binary search in Python.",
        "Write a function to reverse a linked list.",
        "How do you find the longest common subsequence of two strings?",
        "Write Python code to flatten a nested list.",
        "Implement a stack using two queues.",
        "Write a function to check if a string is a valid palindrome, ignoring spaces and punctuation.",
        "How do you detect a cycle in a linked list?",
    ],
    "creative": [
        "Write a short story about a robot learning to paint.",
        "Describe a sunset on Mars in vivid detail.",
        "Write a limerick about a forgetful professor.",
        "Imagine you're a medieval knight encountering a smartphone. Describe your reaction.",
        "Write a dialogue between the Sun and the Moon.",
        "Describe the taste of the color blue.",
        "Write the opening paragraph of a mystery novel set in a library.",
        "Create a recipe for an impossible dish: cloud soup.",
    ],
}


def build_duplicated_forward(model, start, end):
    """
    Build a function that runs the model with layers [start, end) duplicated.
    Returns (h_before_block, h_after_first_pass, h_after_second_pass) at the
    block boundaries for computing spectral metrics.
    """
    inner = model.model

    def run_with_duplication(input_ids, attention_mask):
        # Get embeddings
        h = inner.embed_tokens(input_ids)
        seq_len = h.shape[1]

        # Compute position embeddings once (rotary cos/sin)
        position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0)
        pos_embeds = inner.rotary_emb(h, position_ids)

        # Run layers before the block
        for i in range(start):
            out = inner.layers[i](h, position_embeddings=pos_embeds, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out

        h_before = h.detach().clone()

        # First pass through block
        h1 = h
        for i in range(start, end):
            out = inner.layers[i](h1, position_embeddings=pos_embeds, use_cache=False)
            h1 = out[0] if isinstance(out, tuple) else out

        h_after_first = h1.detach().clone()

        # Second pass through block (same layers)
        h2 = h1
        for i in range(start, end):
            out = inner.layers[i](h2, position_embeddings=pos_embeds, use_cache=False)
            h2 = out[0] if isinstance(out, tuple) else out

        h_after_second = h2.detach().clone()

        return h_before, h_after_first, h_after_second

    return run_with_duplication


def compute_esr_score(h_before, h_after_first, h_after_second,
                      lm_head=None, norm=None):
    """
    Compute Exact Spectral Router score for a block.

    V2 scoring: uses LM-head margin gain as output-quality signal.
    Pure geometric metrics (rho, residual) are misleading because
    destructive blocks show low rho + high residual (looks like
    "good contraction" but is actually "signal destroyed").

    The margin gain (top-1 logit - top-2 logit) directly measures
    whether the second pass makes the model MORE confident.
    """
    def flat_norm(t):
        return t.float().reshape(t.shape[0], -1).norm(dim=-1)

    def top2_margin(h):
        if lm_head is None or norm is None:
            return torch.tensor(0.0, device=h.device)
        h_normed = norm(h[:, -1:, :])
        logits = lm_head(h_normed)[:, -1, :]
        vals = logits.float().topk(k=2, dim=-1).values
        return (vals[:, 0] - vals[:, 1]).mean()

    d1 = flat_norm(h_after_first - h_before)
    d2 = flat_norm(h_after_second - h_after_first)

    rho = (d2 / (d1 + 1e-6)).item()
    residual = torch.log1p(d2).item()

    # Margin gain: does the second pass increase confidence?
    margin_before = top2_margin(h_after_first)
    margin_after = top2_margin(h_after_second)
    margin_gain = (margin_after - margin_before).item()

    # V2 scoring: margin gain is primary, contraction is secondary
    # Positive margin_gain = second pass helps. Negative = it hurts.
    score = 0.50 * margin_gain + 0.30 * (1.0 - rho) + 0.20 * min(residual, 6.0)
    return score, rho, residual, margin_gain


def run_diagnostic(model_path, candidate_blocks, prompt_bank):
    """Run the full diagnostic."""
    print(f"\n{'='*70}")
    print("ROUTING DIAGNOSTIC: Per-Input vs Per-Task Block Selection")
    print(f"{'='*70}")

    model, tokenizer = load_original_model(model_path)
    device = next(model.parameters()).device

    # Get LM head and final norm for margin scoring
    lm_head = model.lm_head
    final_norm = model.model.norm

    # Build duplicated forwards for each candidate
    block_runners = {}
    for block in candidate_blocks:
        name = block["name"]
        block_runners[name] = build_duplicated_forward(
            model, block["start"], block["end"]
        )
        print(f"  Block '{name}': layers ({block['start']},{block['end']})")

    block_names = [b["name"] for b in candidate_blocks]
    K = len(block_names)

    # Collect ESR scores for every (task, prompt, block) triple
    all_results = []
    task_scores = defaultdict(list)  # task -> list of score vectors
    task_winners = defaultdict(list)  # task -> list of winning block indices

    total_prompts = sum(len(v) for v in prompt_bank.items())
    done = 0

    for task_name, prompts in prompt_bank.items():
        print(f"\n  Task: {task_name} ({len(prompts)} prompts)")

        for prompt in prompts:
            inp = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=64).to(device)

            scores = []
            rhos = []
            residuals = []
            margins = []

            for block in candidate_blocks:
                name = block["name"]
                runner = block_runners[name]

                with torch.no_grad():
                    h_before, h_first, h_second = runner(
                        inp['input_ids'], inp['attention_mask']
                    )

                score, rho, residual, margin_gain = compute_esr_score(
                    h_before, h_first, h_second,
                    lm_head=lm_head, norm=final_norm
                )
                scores.append(score)
                rhos.append(rho)
                residuals.append(residual)
                margins.append(margin_gain)

            score_vec = np.array(scores)
            winner = int(np.argmax(score_vec))

            task_scores[task_name].append(score_vec)
            task_winners[task_name].append(winner)

            all_results.append({
                "task": task_name,
                "prompt": prompt[:80],
                "scores": scores,
                "rhos": rhos,
                "residuals": residuals,
                "margin_gains": margins,
                "winner": block_names[winner],
                "winner_idx": winner,
            })

            done += 1
            if done % 10 == 0:
                print(f"    [{done}/{total_prompts}] Last winner: {block_names[winner]} "
                      f"(margins: {[f'{m:.2f}' for m in margins]})")

    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    # 1. Per-task dominance
    print("\n--- Per-Task Block Winner Distribution ---")
    for task_name in prompt_bank:
        winners = task_winners[task_name]
        counts = defaultdict(int)
        for w in winners:
            counts[block_names[w]] += 1
        total = len(winners)
        dominant = max(counts.values()) / total
        print(f"  {task_name:25s}: ", end="")
        for name in block_names:
            pct = counts[name] / total * 100
            print(f"{name}={pct:.0f}% ", end="")
        print(f"  [dominance={dominant:.0%}]")

    # 2. Conditional entropy H(B* | T)
    print("\n--- Conditional Entropy H(B* | T) ---")
    total_entropy = 0
    total_prompts_count = 0
    for task_name in prompt_bank:
        winners = task_winners[task_name]
        counts = defaultdict(int)
        for w in winners:
            counts[w] += 1
        total = len(winners)
        h = 0
        for c in counts.values():
            p = c / total
            if p > 0:
                h -= p * np.log2(p)
        total_entropy += h * total
        total_prompts_count += total
        print(f"  {task_name:25s}: H(B*|T={task_name}) = {h:.3f} bits")

    cond_entropy = total_entropy / total_prompts_count
    print(f"\n  Overall H(B* | T) = {cond_entropy:.3f} bits")
    print(f"  Max possible H = {np.log2(K):.3f} bits (uniform over {K} blocks)")
    print(f"  Normalized: {cond_entropy / np.log2(K):.1%} of maximum")

    # 3. Marginal entropy H(B*)
    all_winners = []
    for task_name in prompt_bank:
        all_winners.extend(task_winners[task_name])
    marginal_counts = defaultdict(int)
    for w in all_winners:
        marginal_counts[w] += 1
    total = len(all_winners)
    h_marginal = 0
    for c in marginal_counts.values():
        p = c / total
        if p > 0:
            h_marginal -= p * np.log2(p)
    print(f"\n  Marginal H(B*) = {h_marginal:.3f} bits")
    print(f"  Mutual Information I(B*; T) = H(B*) - H(B*|T) = {h_marginal - cond_entropy:.3f} bits")

    # 4. Score vector variance analysis
    print("\n--- Score Vector Variance ---")
    all_score_vecs = []
    for task_name in prompt_bank:
        all_score_vecs.extend(task_scores[task_name])
    all_score_vecs = np.array(all_score_vecs)

    total_var = np.var(all_score_vecs, axis=0).sum()

    between_task_var = 0
    within_task_var = 0
    grand_mean = all_score_vecs.mean(axis=0)
    for task_name in prompt_bank:
        vecs = np.array(task_scores[task_name])
        task_mean = vecs.mean(axis=0)
        between_task_var += len(vecs) * np.sum((task_mean - grand_mean) ** 2)
        within_task_var += np.sum((vecs - task_mean) ** 2)

    between_task_var /= len(all_score_vecs)
    within_task_var /= len(all_score_vecs)

    print(f"  Total variance:       {total_var:.6f}")
    print(f"  Between-task variance: {between_task_var:.6f} ({between_task_var/total_var:.1%})")
    print(f"  Within-task variance:  {within_task_var:.6f} ({within_task_var/total_var:.1%})")

    # 5. Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    # Decision criteria
    task_determined = between_task_var / total_var if total_var > 0 else 0
    avg_dominance = np.mean([
        max(defaultdict(int, {block_names[w]: 1 for w in task_winners[t]}).values())
        / len(task_winners[t])
        for t in prompt_bank
    ])

    if task_determined > 0.85 and cond_entropy / np.log2(K) < 0.15:
        verdict = "TASK CLASSIFIER — block choice is mostly task-determined"
    elif task_determined > 0.60:
        verdict = "HIERARCHICAL — task-level routing with per-input refinement"
    else:
        verdict = "GEOMETRIC ROUTER — significant per-input signal justifies learned routing"

    print(f"\n  Task-determined variance: {task_determined:.1%}")
    print(f"  Conditional entropy (normalized): {cond_entropy / np.log2(K):.1%}")
    print(f"  Recommendation: {verdict}")

    # Save
    output = {
        "candidate_blocks": candidate_blocks,
        "results": all_results,
        "analysis": {
            "conditional_entropy": cond_entropy,
            "marginal_entropy": h_marginal,
            "mutual_information": h_marginal - cond_entropy,
            "between_task_variance_pct": between_task_var / total_var if total_var > 0 else 0,
            "within_task_variance_pct": within_task_var / total_var if total_var > 0 else 0,
            "verdict": verdict,
        }
    }

    with open(RESULTS_DIR / "diagnostic_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {RESULTS_DIR / 'diagnostic_results.json'}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return output


def main():
    M7B = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"
    run_diagnostic(M7B, CANDIDATE_BLOCKS_7B, PROMPT_BANK)


if __name__ == "__main__":
    main()
