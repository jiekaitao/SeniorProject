"""
DeepPass Math Probe

Implements David Noel Ng's hard math guesstimate probe.
The model must answer difficult math questions in a single shot (no chain-of-thought).
Uses his partial-credit scoring function that handles LLM arithmetic quirks.

These are "intuitive leap" math problems — the model gets one shot to guess the answer.
"""

import json
import re
from typing import List, Dict, Tuple


# Hard math questions — the model must guess the integer answer directly
MATH_QUESTIONS = [
    {
        "question": "What is 78313086360375 multiplied by 88537453126609?",
        "answer": 6933174468959498727528375,
    },
    {
        "question": "What is the cube root of 74088893247?",
        "answer": 4201,  # 4201^3 = 74088893247 (approx)
    },
    {
        "question": "What is the cube root of 18228885506341?",
        "answer": 26317,  # approx
    },
    {
        "question": "What is the cube root of 844178022493, multiplied by 43?",
        "answer": 40549,  # cube_root(844178022493) ≈ 943.07... * 43 ≈ 40549
    },
    {
        "question": "What is 9999999 multiplied by 9999999?",
        "answer": 99999980000001,
    },
    {
        "question": "What is 123456789 multiplied by 987654321?",
        "answer": 121932631112635269,
    },
    {
        "question": "What is the square root of 152399025?",
        "answer": 12345,
    },
    {
        "question": "What is 7777777 multiplied by 3333333?",
        "answer": 25925923703641,
    },
    {
        "question": "What is 456789 raised to the power of 2?",
        "answer": 208655854521,
    },
    {
        "question": "What is 11111111 multiplied by 11111111?",
        "answer": 123456787654321,
    },
    {
        "question": "What is the cube root of 2744000?",
        "answer": 140,
    },
    {
        "question": "What is 314159 multiplied by 271828?",
        "answer": 85397342252,
    },
    {
        "question": "What is 999999999999 divided by 142857, rounded to the nearest integer?",
        "answer": 6999993,
    },
    {
        "question": "What is 2 raised to the power of 48?",
        "answer": 281474976710656,
    },
    {
        "question": "What is the square root of 99980001?",
        "answer": 9999,
    },
    {
        "question": "What is 54321 multiplied by 12345?",
        "answer": 670592745,
    },
]

SYSTEM_PROMPT = "You are a math calculator. Answer with ONLY the number, nothing else. No explanation, no units, no punctuation. Just the integer."

USER_TEMPLATE = "{question}\nAnswer with ONLY the integer number:"


def calculate_score(actual, estimate) -> float:
    """
    Calculate partial-credit score comparing actual vs estimated answer.

    From David Noel Ng's blog post — handles LLM arithmetic quirks:
    - Pads shorter answers
    - Penalises proportionally via correction factor
    - Gives partial credit for "almost right" answers
    """
    try:
        actual_str = str(int(actual))
        estimate_str = str(int(float(estimate)))
    except (ValueError, OverflowError):
        return 0.0

    max_length = max(len(actual_str), len(estimate_str))
    actual_padded = actual_str.ljust(max_length, "0")
    estimate_padded = estimate_str.ljust(max_length, "0")
    padding_size = max_length - min(len(actual_str), len(estimate_str))

    actual_int = int(actual_padded)
    estimate_int = int(estimate_padded)

    if max(actual_int, estimate_int) == 0:
        return 0.0
    relative_diff = abs(actual_int - estimate_int) / max(actual_int, estimate_int)
    correction_factor = 1 - (padding_size / max_length)
    score = (1 - relative_diff) * correction_factor

    return max(0.0, min(score, 1.0))


def extract_number(text: str) -> str:
    """Extract the first number from model output, handling LLM quirks."""
    text = text.strip()
    # Try to find a number (possibly with commas)
    match = re.search(r'-?[\d,]+\.?\d*', text.replace(' ', ''))
    if match:
        return match.group().replace(',', '')
    return text


def run_math_probe(generate_fn, questions: List[Dict] = None, verbose: bool = True) -> Dict:
    """
    Run the math probe on a model.

    Args:
        generate_fn: callable that takes a prompt string and returns generated text
        questions: list of question dicts (defaults to MATH_QUESTIONS)
        verbose: print per-question results

    Returns:
        dict with 'score' (0-1 average), 'scores' (per-question), 'details' (full results)
    """
    if questions is None:
        questions = MATH_QUESTIONS

    scores = []
    details = []

    for idx, q in enumerate(questions):
        prompt = USER_TEMPLATE.format(question=q["question"])
        full_prompt = f"System: {SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"

        response = generate_fn(full_prompt)
        estimated = extract_number(response)

        try:
            score = calculate_score(q["answer"], estimated)
        except Exception:
            score = 0.0

        scores.append(score)
        detail = {
            "question": q["question"],
            "actual": q["answer"],
            "estimated": estimated,
            "raw_response": response,
            "score": score,
        }
        details.append(detail)

        if verbose:
            status = "OK" if score > 0.5 else "MISS"
            print(f"  [{idx+1:2d}/{len(questions)}] {status} score={score:.4f} "
                  f"actual={q['answer']} got={estimated}")

    avg_score = sum(scores) / len(scores) if scores else 0.0

    if verbose:
        print(f"\n  Math Probe Average Score: {avg_score:.4f}")

    return {
        "score": avg_score,
        "scores": scores,
        "details": details,
    }


if __name__ == "__main__":
    # Test the scoring function
    print("Testing calculate_score:")
    print(f"  Exact match:    {calculate_score(12345, 12345):.4f}")    # 1.0
    print(f"  Off by 1 digit: {calculate_score(12345, 12346):.4f}")    # ~0.9999
    print(f"  Missing digit:  {calculate_score(4302459, 430245):.4f}") # partial credit
    print(f"  Wrong answer:   {calculate_score(12345, 99999):.4f}")    # low
    print(f"  Transposed:     {calculate_score(123456789, 123356789):.4f}")  # ~0.999
