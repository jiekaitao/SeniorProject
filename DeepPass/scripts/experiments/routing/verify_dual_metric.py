"""
Verify: Does using a broader eval metric (not just math probe) change the
ranking of spectral candidates?

Tests top-10 brain scanner configs on BOTH:
1. Math probe (Ng's narrow metric)
2. A diverse eval (reasoning + instruction following + knowledge)

If the ranking changes, it demonstrates that spectral screening finds the
right REGION, and the final selection depends on the eval metric.
"""

import sys, os, json, time, torch, gc
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts'))

from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe

RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/dual_metric_verification")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Diverse eval — questions with clear correct answers across task types
DIVERSE_EVAL = [
    # Reasoning (Ng used EQ-bench which tests emotional reasoning)
    {"q": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost? Answer with just the number in cents.",
     "correct": ["5", "five", "0.05", "$0.05", "5 cents"], "type": "reasoning"},
    {"q": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Answer with just the number of minutes.",
     "correct": ["5", "five"], "type": "reasoning"},
    {"q": "A farmer has 17 sheep. All but 9 die. How many are left? Answer with just the number.",
     "correct": ["9", "nine"], "type": "reasoning"},
    {"q": "If you overtake the person in 2nd place in a race, what place are you in? Answer with just the number.",
     "correct": ["2", "2nd", "second"], "type": "reasoning"},
    {"q": "You have 8 balls, one is heavier. Using a balance scale, what is the minimum number of weighings to find the heavy ball? Answer with just the number.",
     "correct": ["2", "two"], "type": "reasoning"},

    # Instruction following
    {"q": "Respond with ONLY a single word that means 'happy'.",
     "correct": ["joyful", "glad", "cheerful", "content", "delighted", "elated", "pleased", "merry", "jovial", "blissful", "ecstatic"], "type": "instruction"},
    {"q": "Name exactly three countries that start with the letter B, separated by commas.",
     "correct": ["brazil", "belgium", "bhutan", "bolivia", "botswana", "brunei", "bulgaria", "burkina", "burundi", "bahamas", "bahrain", "bangladesh", "barbados", "belarus", "belize", "benin"], "type": "instruction", "check": "count_b_countries"},
    {"q": "What is 7 times 8? Respond with ONLY the number, nothing else.",
     "correct": ["56"], "type": "instruction"},

    # Knowledge
    {"q": "What is the chemical symbol for gold? Answer with just the symbol.",
     "correct": ["au", "Au", "AU"], "type": "knowledge"},
    {"q": "In what year did World War 2 end? Answer with just the year.",
     "correct": ["1945"], "type": "knowledge"},
    {"q": "What is the capital of Japan? Answer with just the city name.",
     "correct": ["tokyo", "Tokyo"], "type": "knowledge"},
    {"q": "How many planets are in our solar system? Answer with just the number.",
     "correct": ["8", "eight"], "type": "knowledge"},
]


def score_diverse_eval(generate_fn):
    """Score model on diverse eval. Returns accuracy 0-1."""
    correct = 0
    total = len(DIVERSE_EVAL)

    for item in DIVERSE_EVAL:
        prompt = f"Question: {item['q']}\nAnswer:"
        response = generate_fn(prompt).strip().lower()

        # Check if any correct answer appears in response
        is_correct = False

        if item.get("check") == "count_b_countries":
            # Special: check that response has 3 items starting with B
            parts = [p.strip().lower() for p in response.replace('\n', ',').split(',')]
            b_countries = [p for p in parts if p and p[0] == 'b']
            is_correct = len(b_countries) >= 3
        else:
            for ans in item["correct"]:
                if ans.lower() in response:
                    is_correct = True
                    break

        if is_correct:
            correct += 1

    return correct / total


def _run_layer_range(inner, h, start, end, pos_embeds):
    for i in range(start, end):
        out = inner.layers[i](h, position_embeddings=pos_embeds, use_cache=False)
        h = out[0] if isinstance(out, tuple) else out
    return h


def generate_with_dup(model, tokenizer, prompt, start, end, max_new_tokens=64):
    """Generate with layers [start, end) duplicated."""
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

        h = _run_layer_range(inner, h, 0, start, pos_embeds)
        h = _run_layer_range(inner, h, start, end, pos_embeds)
        h = _run_layer_range(inner, h, start, end, pos_embeds)  # duplication
        h = _run_layer_range(inner, h, end, N, pos_embeds)
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


def main():
    model_path = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"

    # Load brain scanner results to get top configs
    with open("/blue/cis4914/jietao/DeepPass/results/sweep_7B/sweep_results.json") as f:
        sweep = json.load(f)

    baseline_score = sweep["baseline_score"]
    configs = []
    for key, val in sweep["results"].items():
        i, j = [int(x) for x in key.split(",")]
        configs.append({"i": i, "j": j, "math_delta": val["delta"],
                        "math_score": val["score"]})

    # Sort by math probe delta, take top 10
    configs.sort(key=lambda x: -x["math_delta"])
    top_configs = configs[:10]

    print(f"{'='*70}")
    print("DUAL METRIC VERIFICATION")
    print(f"Testing top 10 brain scanner configs on math probe + diverse eval")
    print(f"{'='*70}")
    print(f"\nTop 10 by math probe (from brain scanner):")
    for c in top_configs:
        print(f"  ({c['i']},{c['j']}): math_delta={c['math_delta']:+.4f}")

    # Load model
    print(f"\nLoading model...")
    model, tokenizer = load_original_model(model_path)

    # Baseline
    print(f"\n--- Baseline ---")
    def gen_baseline(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    diverse_baseline = score_diverse_eval(gen_baseline)
    print(f"  Math probe: {baseline_score:.4f}")
    print(f"  Diverse eval: {diverse_baseline:.2%}")

    # Test each config
    results = []
    for idx, c in enumerate(top_configs):
        i, j = c["i"], c["j"]
        print(f"\n--- Config ({i},{j}) [{idx+1}/10] ---")

        def gen_dup(prompt, _i=i, _j=j):
            return generate_with_dup(model, tokenizer, prompt, _i, _j)

        # Math probe (we already have this from brain scanner)
        math_score = c["math_score"]
        math_delta = c["math_delta"]

        # Diverse eval (NEW)
        diverse_score = score_diverse_eval(gen_dup)
        diverse_delta = diverse_score - diverse_baseline

        print(f"  Math:    {math_score:.4f} ({math_delta:+.4f})")
        print(f"  Diverse: {diverse_score:.2%} ({diverse_delta:+.2%})")

        # Combined score (equally weighted, like Ng's dual metric)
        combined = 0.5 * (math_delta / 0.26) + 0.5 * (diverse_delta / max(diverse_baseline, 0.01))
        print(f"  Combined: {combined:+.4f}")

        results.append({
            "config": [i, j],
            "math_score": math_score,
            "math_delta": math_delta,
            "diverse_score": diverse_score,
            "diverse_delta": diverse_delta,
            "combined": combined,
        })

    # Rankings
    print(f"\n{'='*70}")
    print("RANKING COMPARISON")
    print(f"{'='*70}")

    by_math = sorted(results, key=lambda x: -x["math_delta"])
    by_diverse = sorted(results, key=lambda x: -x["diverse_delta"])
    by_combined = sorted(results, key=lambda x: -x["combined"])

    print(f"\n{'Rank':>4} {'Math Probe Best':>20} {'Diverse Best':>20} {'Combined Best':>20}")
    print(f"{'':>4} {'(narrow metric)':>20} {'(broad metric)':>20} {'(Ng-style dual)':>20}")
    print("-" * 70)
    for rank in range(min(5, len(results))):
        m = by_math[rank]
        d = by_diverse[rank]
        c = by_combined[rank]
        print(f"{rank+1:>4} ({m['config'][0]},{m['config'][1]}) {m['math_delta']:+.4f}"
              f"   ({d['config'][0]},{d['config'][1]}) {d['diverse_delta']:+.2%}"
              f"   ({c['config'][0]},{c['config'][1]}) {c['combined']:+.4f}")

    print(f"\nDo the rankings change? ", end="")
    math_top = tuple(by_math[0]["config"])
    combined_top = tuple(by_combined[0]["config"])
    if math_top != combined_top:
        print(f"YES — math picks {math_top}, combined picks {combined_top}")
    else:
        print(f"NO — both pick {math_top}")

    # Save
    output = {
        "baseline_math": baseline_score,
        "baseline_diverse": diverse_baseline,
        "results": results,
        "ranking_math": [r["config"] for r in by_math],
        "ranking_diverse": [r["config"] for r in by_diverse],
        "ranking_combined": [r["config"] for r in by_combined],
        "rankings_differ": math_top != combined_top,
    }
    with open(RESULTS_DIR / "dual_metric_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
