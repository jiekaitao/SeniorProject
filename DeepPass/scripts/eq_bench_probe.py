"""
DeepPass EQ-Bench Probe

Lightweight standalone probe for EQ-Bench (emotional intelligence benchmark).
Uses 20 representative questions from the 171-question EQ-Bench v2.1 dataset,
with the same sigmoid scoring function as the official lm-eval implementation.

This runs in ~60s on 7B vs ~10 min for full EQ-Bench via lm-eval, making it
suitable for rapid config screening alongside math_probe.

Ng found that math_probe + EQ-Bench together are a good dual-metric proxy
for broader model capabilities.
"""

import math
import re
from typing import List, Dict, Callable

# 20 evenly-spaced question indices from the 171-question validation set
QUESTION_INDICES = [0, 8, 17, 26, 35, 44, 53, 62, 71, 80,
                    89, 98, 107, 116, 125, 134, 143, 152, 161, 170]


def _load_questions():
    """Load EQ-Bench questions from HuggingFace dataset (cached after first call)."""
    import ast
    from datasets import load_dataset

    ds = load_dataset("pbevan11/EQ-Bench", split="validation")
    questions = []
    for idx in QUESTION_INDICES:
        ref = ast.literal_eval(ds[idx]["reference_answer_fullscale"])
        questions.append(
            {
                "prompt": ds[idx]["prompt"],
                "reference": {
                    ref["emotion1"]: int(ref["emotion1_score"]),
                    ref["emotion2"]: int(ref["emotion2_score"]),
                    ref["emotion3"]: int(ref["emotion3_score"]),
                    ref["emotion4"]: int(ref["emotion4_score"]),
                },
                "index": idx,
            }
        )
    return questions


def calculate_eq_score(reference: Dict[str, int], response_text: str) -> Dict:
    """
    Score a model response against reference emotions.

    Uses the same sigmoid scoring function as the official EQ-Bench lm-eval task.
    Score is on a 0-100 scale (100 = perfect match).

    Returns:
        dict with 'score' (0-100), 'parsed' (bool), 'parsed_emotions' (dict)
    """
    # Parse emotion:score pairs from response
    parsed = dict(re.findall(r"(\w+):\s+(\d+)", response_text))

    if len(parsed) != 4:
        return {"score": 0.0, "parsed": False, "parsed_emotions": parsed}

    # Check that parsed emotions match reference emotions
    matched = {}
    for emotion in parsed:
        if emotion in reference:
            matched[emotion] = True
    if len(matched) != 4:
        return {"score": 0.0, "parsed": False, "parsed_emotions": parsed}

    # Calculate difference tally with sigmoid scaling
    difference_tally = 0.0
    for emotion, user_score_str in parsed.items():
        if emotion in reference:
            d = abs(float(user_score_str) - float(reference[emotion]))
            if d == 0:
                scaled_diff = 0
            elif d <= 5:
                # S-shaped sigmoid: 6.5 / (1 + e^(-1.2*(d-4)))
                scaled_diff = 6.5 * (1 / (1 + math.e ** (-1.2 * (d - 4))))
            else:
                scaled_diff = d
            difference_tally += scaled_diff

    # Invert so higher = better, with adjustment constant for zero-baseline
    adjust_const = 0.7477
    final_score = (10 - (difference_tally * adjust_const)) * 10

    return {
        "score": max(0.0, min(100.0, final_score)),
        "parsed": True,
        "parsed_emotions": parsed,
    }


def run_eq_bench_probe(
    generate_fn: Callable, questions: List[Dict] = None, verbose: bool = True
) -> Dict:
    """
    Run the EQ-Bench probe on a model.

    Args:
        generate_fn: callable that takes a prompt string and returns generated text
        questions: list of question dicts (defaults to 20-question subset)
        verbose: print per-question results

    Returns:
        dict with 'score' (0-100 average), 'scores', 'details', 'parse_rate'
    """
    if questions is None:
        questions = _load_questions()

    scores = []
    details = []
    parsed_count = 0

    for idx, q in enumerate(questions):
        response = generate_fn(q["prompt"])
        result = calculate_eq_score(q["reference"], response)

        scores.append(result["score"])
        if result["parsed"]:
            parsed_count += 1

        detail = {
            "index": q.get("index", idx),
            "reference": q["reference"],
            "raw_response": response[:200],
            "parsed_emotions": result["parsed_emotions"],
            "parsed": result["parsed"],
            "score": result["score"],
        }
        details.append(detail)

        if verbose:
            status = "OK" if result["parsed"] else "FAIL"
            print(
                f"  [{idx+1:2d}/{len(questions)}] {status} "
                f"score={result['score']:.1f}/100 "
                f"emotions={list(q['reference'].keys())}"
            )

    avg_score = sum(scores) / len(scores) if scores else 0.0
    parse_rate = parsed_count / len(questions) if questions else 0.0

    if verbose:
        print(f"\n  EQ-Bench Probe Score: {avg_score:.1f}/100")
        print(f"  Parse rate: {parsed_count}/{len(questions)} ({parse_rate*100:.0f}%)")

    return {
        "score": avg_score,
        "scores": scores,
        "details": details,
        "parse_rate": parse_rate,
        "num_questions": len(questions),
    }


if __name__ == "__main__":
    # Test the scoring function
    print("Testing calculate_eq_score:")

    ref = {"Remorseful": 0, "Indifferent": 6, "Affectionate": 0, "Annoyed": 7}

    # Perfect match
    perfect = "Remorseful: 0\nIndifferent: 6\nAffectionate: 0\nAnnoyed: 7"
    r = calculate_eq_score(ref, perfect)
    print(f"  Perfect match:  {r['score']:.1f}/100 parsed={r['parsed']}")

    # Close match
    close = "Remorseful: 1\nIndifferent: 5\nAffectionate: 1\nAnnoyed: 6"
    r = calculate_eq_score(ref, close)
    print(f"  Close match:    {r['score']:.1f}/100 parsed={r['parsed']}")

    # Bad match
    bad = "Remorseful: 10\nIndifferent: 0\nAffectionate: 10\nAnnoyed: 0"
    r = calculate_eq_score(ref, bad)
    print(f"  Bad match:      {r['score']:.1f}/100 parsed={r['parsed']}")

    # Unparseable
    r = calculate_eq_score(ref, "I think the character feels sad.")
    print(f"  Unparseable:    {r['score']:.1f}/100 parsed={r['parsed']}")
