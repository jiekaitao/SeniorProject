from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict, List, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)

GENERATE_SYSTEM_PROMPT = """You are a training data generator for grid-based reasoning models.

Generate exactly {num_examples} input/output grid pairs for the described problem type.
Each pair must have:
- "input": a 2D array of non-negative integers
- "output": a 2D array of non-negative integers
Grids should be small (2x2 to 6x6) and varied.

Return ONLY a valid JSON array. No markdown, no explanation, no code fences.
Example: [{{"input": [[1,0],[0,1]], "output": [[0,1],[1,0]]}}]"""


def _generate_fallback_examples(num_examples: int) -> List[Dict[str, Any]]:
    """Generate simple identity/transpose examples as fallback."""
    examples = []
    for _ in range(num_examples):
        size = random.randint(2, 5)
        grid = [[random.randint(0, 9) for _ in range(size)] for _ in range(size)]
        # Alternate between identity and transpose
        if random.random() < 0.5:
            output = [row[:] for row in grid]
        else:
            output = [[grid[r][c] for r in range(size)] for c in range(size)]
        examples.append({"input": grid, "output": output})
    return examples


# ---------------------------------------------------------------------------
#  Towers of Hanoi generator (deterministic, no LLM required)
# ---------------------------------------------------------------------------

def _pegs_to_grid(pegs: List[List[int]], height: int) -> List[List[int]]:
    grid: List[List[int]] = []
    for row in range(height):
        r = []
        for peg in pegs:
            r.append(peg[row] if row < len(peg) else 0)
        grid.append(r)
    return list(reversed(grid))


def _solve_hanoi(n: int, source: int = 0, target: int = 2, auxiliary: int = 1) -> List[List[List[int]]]:
    pegs: List[List[int]] = [list(range(n, 0, -1)), [], []]
    states = [_pegs_to_grid(pegs, n)]

    def move(k: int, src: int, tgt: int, aux: int) -> None:
        if k == 0:
            return
        move(k - 1, src, aux, tgt)
        disk = pegs[src].pop()
        pegs[tgt].append(disk)
        states.append(_pegs_to_grid(pegs, n))
        move(k - 1, aux, tgt, src)

    move(n, source, target, auxiliary)
    return states


def _pad_grid(grid: List[List[int]], rows: int, cols: int) -> List[List[int]]:
    out: List[List[int]] = []
    for r in range(rows):
        if r < len(grid):
            row = list(grid[r]) + [0] * (cols - len(grid[r]))
        else:
            row = [0] * cols
        out.append(row[:cols])
    return out


def _generate_hanoi_examples(
    num_examples: int, max_disks: int = 5
) -> List[Dict[str, Any]]:
    """Generate Towers of Hanoi move pairs: input state -> next state.

    Uses the classic recursive optimal solution and enumerates consecutive
    states. Grid is padded to max_disks rows by 3 columns (one per peg).
    """
    pairs: List[Dict[str, Any]] = []
    # Deterministic across disk sizes 2..max_disks.
    for n_disks in range(2, max_disks + 1):
        states = _solve_hanoi(n_disks)
        for i in range(len(states) - 1):
            inp = _pad_grid(states[i], max_disks, 3)
            out = _pad_grid(states[i + 1], max_disks, 3)
            pairs.append({"input": inp, "output": out})

    # Trim (or repeat) to hit the requested count.
    if num_examples <= 0 or not pairs:
        return pairs
    if len(pairs) >= num_examples:
        return pairs[:num_examples]
    extra = []
    i = 0
    while len(pairs) + len(extra) < num_examples:
        extra.append(pairs[i % len(pairs)])
        i += 1
    return pairs + extra


_HANOI_PATTERNS = ("hanoi", "tower of ", "towers of ", "peg")


async def generate_training_data(
    problem_description: str,
    classification: str | None = None,
    num_examples: int = 10,
) -> Tuple[List[Dict[str, Any]], str | None]:
    """Generate training data using LLM based on problem description.

    Returns (grid_pairs, error_message). On success error is None.
    """
    desc_lower = (problem_description or "").lower()
    is_hanoi = any(p in desc_lower for p in _HANOI_PATTERNS)

    if is_hanoi:
        # Hanoi has a closed-form optimal sequence — always prefer the
        # deterministic generator over the LLM so the model gets clean data.
        # 5 disks yields 31 pairs which is plenty for a ~3 min demo.
        return _generate_hanoi_examples(num_examples=max(num_examples, 31)), None

    if not settings.openai_api_key:
        logger.info("No API key — using fallback data generation")
        return _generate_fallback_examples(num_examples), None

    context = problem_description
    if classification:
        context = f"Problem type: {classification}\nDescription: {problem_description}"

    try:
        from services.llm import structured_completion

        result = await structured_completion(
            system_prompt=GENERATE_SYSTEM_PROMPT.format(num_examples=num_examples),
            user_content=context,
            temperature=0.9,
            max_tokens=4000,
        )

        # Strip markdown fences if present
        text = result.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        parsed = json.loads(text)

        if not isinstance(parsed, list) or len(parsed) == 0:
            logger.warning("LLM returned empty/invalid array, using fallback")
            return _generate_fallback_examples(num_examples), None

        # Validate each pair
        valid = []
        for item in parsed:
            if (
                isinstance(item, dict)
                and "input" in item
                and "output" in item
                and isinstance(item["input"], list)
                and isinstance(item["output"], list)
            ):
                valid.append({"input": item["input"], "output": item["output"]})

        if len(valid) == 0:
            return _generate_fallback_examples(num_examples), None

        return valid, None

    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON for data generation, using fallback")
        return _generate_fallback_examples(num_examples), None
    except Exception as e:
        logger.warning("Data generation via LLM failed, using fallback", exc_info=True)
        return _generate_fallback_examples(num_examples), None
