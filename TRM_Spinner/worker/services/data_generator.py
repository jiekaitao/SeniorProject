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


async def generate_training_data(
    problem_description: str,
    classification: str | None = None,
    num_examples: int = 10,
) -> Tuple[List[Dict[str, Any]], str | None]:
    """Generate training data using LLM based on problem description.

    Returns (grid_pairs, error_message). On success error is None.
    """
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
