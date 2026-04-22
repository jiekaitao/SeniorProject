"""
Reasoning Probe: Tests "think twice" hypothesis on trick questions

Questions that require careful multi-step reasoning where LLMs typically fail
on first pass but succeed with either:
  1. Prompt duplication (repeating the question in input)
  2. Layer duplication (K=2+ through core blocks)
  3. PSRT recursion (iterating reasoning channel)

Tests the hypothesis: "attention re-computation gives the model a second
chance to properly attend to key tokens it missed on the first pass."

Usage:
    python reasoning_probe.py --model <path> --name <name> --core_start 10 --core_end 13
"""

import os, sys, json, time, argparse, re
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from layer_duplicator import load_original_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Trick reasoning questions
# Each has: question, correct_answer, wrong_answer, reasoning
# The model must demonstrate understanding, not just pattern match
# ============================================================

QUESTIONS = [
    {
        "id": "car_wash",
        "question": "I live 100 meters away from the car wash. I want to wash my car. Should I drive or walk?",
        "correct": "drive",
        "wrong": "walk",
        "reasoning": "You need your car AT the car wash to wash it, so you must drive.",
    },
    {
        "id": "umbrella",
        "question": "I'm carrying a large painting that can't get wet. It's raining outside. I have an umbrella. Should I use the umbrella or put the painting under my coat?",
        "correct": "umbrella",
        "wrong": "coat",
        "reasoning": "The painting is large - it won't fit under a coat. Use the umbrella to cover it.",
    },
    {
        "id": "elevator",
        "question": "A man lives on the 10th floor. Every day he takes the elevator down to the ground floor to go to work. When he comes back, he takes the elevator to the 7th floor and walks up 3 flights of stairs. Why?",
        "correct": "short",  # he's too short to reach the 10th floor button
        "wrong": "exercise",
        "reasoning": "He's too short to reach the button for the 10th floor, can only reach 7.",
    },
    {
        "id": "surgeon",
        "question": "A father and son are in a car accident. The father dies. The son is rushed to the hospital. The surgeon says 'I can't operate on this boy, he's my son.' How is this possible?",
        "correct": "mother",
        "wrong": "impossible",
        "reasoning": "The surgeon is the boy's mother.",
    },
    {
        "id": "coins",
        "question": "I have two coins that add up to 30 cents. One of them is not a nickel. What are the two coins?",
        "correct": "quarter",  # a quarter and a nickel (ONE of them is not a nickel - the other is)
        "wrong": "dime",
        "reasoning": "A quarter (25c) and a nickel (5c). One of them (the quarter) is not a nickel.",
    },
    {
        "id": "bus_driver",
        "question": "You are a bus driver. At the first stop, 4 people get on. At the second stop, 8 people get on and 2 get off. At the third stop, 3 people get on and 5 get off. What color are the bus driver's eyes?",
        "correct": "your",  # YOUR eye color - you are the bus driver
        "wrong": "number",
        "reasoning": "The question says 'You are a bus driver' - the answer is your eye color.",
    },
    {
        "id": "rope",
        "question": "A man is found dead in a room with 53 bicycles. How did he die?",
        "correct": "cheat",  # playing cards - bicycle is a brand of cards, he was caught cheating
        "wrong": "crushed",
        "reasoning": "Bicycle is a brand of playing cards. He was caught cheating at poker (53 cards = extra card).",
    },
    {
        "id": "time_zones",
        "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
        "correct": "5",  # still 5 minutes
        "wrong": "100",
        "reasoning": "Each machine makes 1 widget in 5 minutes. 100 machines make 100 widgets in 5 minutes.",
    },
    {
        "id": "lily_pad",
        "question": "A lily pad doubles in size every day. If it takes 48 days for the lily pad to cover the entire lake, how many days does it take to cover half the lake?",
        "correct": "47",
        "wrong": "24",
        "reasoning": "If it doubles daily and covers the whole lake on day 48, it covered half on day 47.",
    },
    {
        "id": "bat_ball",
        "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "correct": "5",  # 5 cents
        "wrong": "10",
        "reasoning": "Ball = $0.05, Bat = $1.05. $1.05 - $0.05 = $1.00 more. Total = $1.10.",
    },
    {
        "id": "freezing",
        "question": "What happens to water when it reaches 0 degrees Celsius at standard atmospheric pressure?",
        "correct": "freeze",
        "wrong": "nothing",
        "reasoning": "Water freezes at 0°C / 32°F at standard pressure.",
    },
    {
        "id": "apple_tree",
        "question": "If you have 3 apples and you take away 2, how many apples do you have?",
        "correct": "2",  # YOU took 2, so YOU have 2
        "wrong": "1",
        "reasoning": "You TOOK 2 apples, so YOU have 2 apples.",
    },
]


def score_response(response, question):
    """Score whether the model got the right answer."""
    resp_lower = response.lower()
    correct_key = question["correct"].lower()
    wrong_key = question["wrong"].lower()

    # Special scoring per question
    qid = question["id"]

    if qid == "car_wash":
        if "drive" in resp_lower and "walk" not in resp_lower.split("drive")[0][-20:]:
            return 1.0
        if "need" in resp_lower and "car" in resp_lower and "drive" in resp_lower:
            return 1.0
        if "walk" in resp_lower and "drive" not in resp_lower:
            return 0.0
        return 0.5  # ambiguous

    elif qid == "bat_ball":
        if "5 cent" in resp_lower or "$0.05" in resp_lower or "0.05" in resp_lower:
            return 1.0
        if "10 cent" in resp_lower or "$0.10" in resp_lower or "10c" in resp_lower:
            return 0.0
        return 0.5

    elif qid == "lily_pad":
        if "47" in resp_lower:
            return 1.0
        if "24" in resp_lower:
            return 0.0
        return 0.5

    elif qid == "time_zones":
        # Check if they say 5 minutes (not 100)
        if "5 minute" in resp_lower and "100" not in resp_lower:
            return 1.0
        if "100 minute" in resp_lower:
            return 0.0
        if "still 5" in resp_lower or "same" in resp_lower:
            return 1.0
        return 0.5

    elif qid == "surgeon":
        if "mother" in resp_lower or "mom" in resp_lower:
            return 1.0
        if "impossible" in resp_lower or "can't" in resp_lower:
            return 0.0
        return 0.5

    elif qid == "apple_tree":
        # Tricky: answer is 2 (you took them)
        if "2" in resp_lower and "1" not in resp_lower[:resp_lower.index("2")] if "2" in resp_lower else False:
            return 1.0
        if resp_lower.strip().startswith("2") or "you have 2" in resp_lower:
            return 1.0
        if "1" in resp_lower and "2" not in resp_lower:
            return 0.0
        return 0.5

    elif qid == "bus_driver":
        if "your" in resp_lower or "you are" in resp_lower:
            return 1.0
        return 0.0

    # Generic scoring
    if correct_key in resp_lower:
        return 1.0
    if wrong_key in resp_lower and correct_key not in resp_lower:
        return 0.0
    return 0.5


class LayerIdxWrapper(nn.Module):
    def __init__(self, layer, new_idx):
        super().__init__()
        self.layer = layer
        self.new_layer_idx = new_idx
        self.orig_idx = layer.layer_idx
        self.orig_attn = layer.self_attn.layer_idx if hasattr(layer, 'self_attn') else None

    def forward(self, *args, **kwargs):
        self.layer.layer_idx = self.new_layer_idx
        if hasattr(self.layer, 'self_attn'):
            self.layer.self_attn.layer_idx = self.new_layer_idx
        try:
            return self.layer(*args, **kwargs)
        finally:
            self.layer.layer_idx = self.orig_idx
            if self.orig_attn is not None:
                self.layer.self_attn.layer_idx = self.orig_attn

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)


def get_inner(model):
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    return inner


def build_k_recursive(model, core_start, core_end, K):
    inner = get_inner(model)
    original_layers = list(inner.layers)
    N = len(original_layers)
    order = list(range(core_start))
    for _ in range(K):
        order.extend(range(core_start, core_end))
    order.extend(range(core_end, N))

    seen = set()
    new_layers = []
    for pi, oi in enumerate(order):
        l = original_layers[oi]
        if oi in seen:
            new_layers.append(LayerIdxWrapper(l, pi))
        else:
            l.layer_idx = pi
            if hasattr(l, 'self_attn'):
                l.self_attn.layer_idx = pi
            new_layers.append(l)
        seen.add(oi)

    inner.layers = nn.ModuleList(new_layers)
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg is None: continue
        if hasattr(cfg, 'num_hidden_layers'): cfg.num_hidden_layers = len(new_layers)
        if hasattr(cfg, 'layer_types') and cfg.layer_types:
            cfg.layer_types = [cfg.layer_types[oi % len(cfg.layer_types)] for oi in order]
    if hasattr(inner, 'config') and hasattr(inner.config, 'num_hidden_layers'):
        inner.config.num_hidden_layers = len(new_layers)
    return original_layers, N


def restore(model, original_layers, N):
    inner = get_inner(model)
    inner.layers = nn.ModuleList(original_layers)
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg and hasattr(cfg, 'num_hidden_layers'): cfg.num_hidden_layers = N
    if hasattr(inner, 'config') and hasattr(inner.config, 'num_hidden_layers'):
        inner.config.num_hidden_layers = N
    for i, l in enumerate(original_layers):
        l.layer_idx = i
        if hasattr(l, 'self_attn'): l.self_attn.layer_idx = i


def generate(model, tokenizer, prompt, device, max_tokens=150):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    kw = {k: v for k, v in inputs.items() if k != 'token_type_ids'}
    with torch.no_grad():
        out = model.generate(**kw, max_new_tokens=max_tokens, do_sample=False, use_cache=False)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def run_probe(model, tokenizer, device, questions, tag=""):
    """Run all questions, return scores."""
    scores = []
    for q in questions:
        prompt = f"Answer concisely.\n\nQuestion: {q['question']}\n\nAnswer:"
        response = generate(model, tokenizer, prompt, device)
        score = score_response(response, q)
        scores.append(score)
        status = "CORRECT" if score == 1.0 else ("WRONG" if score == 0.0 else "PARTIAL")
        print(f'  [{status}] {q["id"]}: {response[:80].strip()}', flush=True)
    avg = sum(scores) / len(scores) * 100
    print(f'  {tag}: {avg:.1f}% ({sum(1 for s in scores if s==1.0)}/{len(scores)} correct)', flush=True)
    return {'score': avg, 'per_question': {q['id']: s for q, s in zip(questions, scores)}}


def run_prompt_duplication_probe(model, tokenizer, device, questions, tag=""):
    """Test prompt duplication (repeating the question twice in input)."""
    scores = []
    for q in questions:
        # Duplicate the question in the prompt
        prompt = (f"Answer concisely.\n\n"
                  f"Question: {q['question']}\n"
                  f"Question: {q['question']}\n\n"
                  f"Answer:")
        response = generate(model, tokenizer, prompt, device)
        score = score_response(response, q)
        scores.append(score)
        status = "CORRECT" if score == 1.0 else ("WRONG" if score == 0.0 else "PARTIAL")
        print(f'  [{status}] {q["id"]}: {response[:80].strip()}', flush=True)
    avg = sum(scores) / len(scores) * 100
    print(f'  {tag}: {avg:.1f}% ({sum(1 for s in scores if s==1.0)}/{len(scores)} correct)', flush=True)
    return {'score': avg, 'per_question': {q['id']: s for q, s in zip(questions, scores)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--cache_dir', default='/blue/cis4914/jietao/hf_cache')
    parser.add_argument('--core_start', type=int, required=True)
    parser.add_argument('--core_end', type=int, required=True)
    parser.add_argument('--max_k', type=int, default=3)
    args = parser.parse_args()

    SAVE_DIR = f'results/data/reasoning_probe/{args.name}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda')

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, device_map='auto',
        dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.eval()

    results = {}

    # === 1. Baseline K=1 ===
    print(f'\n=== K=1 (baseline) ===', flush=True)
    results['K=1'] = run_probe(model, tokenizer, device, QUESTIONS, 'K=1')

    # === 2. Prompt duplication (no layer dup) ===
    print(f'\n=== Prompt Duplication (question repeated 2x) ===', flush=True)
    results['prompt_dup'] = run_prompt_duplication_probe(model, tokenizer, device, QUESTIONS,
                                                         'prompt_dup')

    # === 3. Layer duplication K=2..max_k ===
    for K in range(2, args.max_k + 1):
        print(f'\n=== Layer Duplication K={K} ===', flush=True)
        orig, N = build_k_recursive(model, args.core_start, args.core_end, K)
        results[f'K={K}'] = run_probe(model, tokenizer, device, QUESTIONS, f'K={K}')
        restore(model, orig, N)

    # === Summary ===
    print(f'\n{"=" * 70}', flush=True)
    print(f'REASONING PROBE SUMMARY -- {args.name}', flush=True)
    print(f'Core: [{args.core_start}, {args.core_end})', flush=True)
    print(f'{"=" * 70}', flush=True)

    print(f'\n{"Config":<20} {"Score":>8}  Per-question breakdown', flush=True)
    print(f'{"-" * 70}', flush=True)
    for config, data in results.items():
        per_q = data['per_question']
        breakdown = ' '.join(f'{v:.0f}' for v in per_q.values())
        print(f'{config:<20} {data["score"]:>7.1f}%  [{breakdown}]', flush=True)

    # Per-question comparison
    print(f'\nPer-question detail:', flush=True)
    for q in QUESTIONS:
        scores_str = ' | '.join(f'{config}: {data["per_question"][q["id"]]:.0f}'
                                for config, data in results.items())
        print(f'  {q["id"]:<15} {scores_str}', flush=True)

    # Did prompt dup help?
    baseline_score = results['K=1']['score']
    prompt_dup_score = results['prompt_dup']['score']
    k2_score = results.get('K=2', {}).get('score', 0)
    print(f'\n  Baseline:       {baseline_score:.1f}%', flush=True)
    print(f'  Prompt dup:     {prompt_dup_score:.1f}% ({prompt_dup_score-baseline_score:+.1f})', flush=True)
    print(f'  Layer dup K=2:  {k2_score:.1f}% ({k2_score-baseline_score:+.1f})', flush=True)
    print(f'  Same mechanism? {"YES" if (prompt_dup_score > baseline_score) == (k2_score > baseline_score) else "MIXED"}', flush=True)
    print('COMPLETE', flush=True)

    with open(f'{SAVE_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved to {SAVE_DIR}/results.json', flush=True)


if __name__ == '__main__':
    main()
