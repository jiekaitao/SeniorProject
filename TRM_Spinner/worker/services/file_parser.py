from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)

PARSE_SYSTEM_PROMPT = """You are a data extraction tool. Given a file's content, extract grid-based input/output pairs.

Each pair must have:
- "input": a 2D array of non-negative integers (list of lists)
- "output": a 2D array of non-negative integers (list of lists)

Return ONLY a valid JSON array of objects. No markdown, no explanation, no code fences.
Example: [{"input": [[1,0],[0,1]], "output": [[0,1],[1,0]]}]

If the data contains CSV columns, text descriptions, or other formats, interpret them as grid pairs.
If the file doesn't contain extractable grid pairs, return an empty array: []"""

# Regex to find JSON arrays inside markdown code fences or bare in text
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n([\s\S]*?)```", re.IGNORECASE)
# Strip common RTF header/footer if someone pasted from ChatGPT via rich text
_RTF_HEADER_RE = re.compile(r"^\{\\rtf[\s\S]*?\\pard\s*", re.IGNORECASE)
_RTF_CONTROL_RE = re.compile(r"\\[a-z]+\d*\s?|\{|\}", re.IGNORECASE)


def _strip_chatgpt_wrapping(text: str) -> str:
    """Strip markdown code fences, RTF wrappers, and prose around JSON.

    Handles common ChatGPT output formats:
    - ```json ... ``` blocks
    - RTF copied from the ChatGPT web UI
    - Prose before/after the JSON array
    """
    stripped = text.strip()

    # Handle RTF: if the file starts with {\rtf, strip control words
    if stripped.startswith("{\\rtf"):
        stripped = _RTF_HEADER_RE.sub("", stripped)
        stripped = _RTF_CONTROL_RE.sub("", stripped)
        stripped = stripped.strip()

    # Try to extract JSON from markdown code fences
    fence_matches = _CODE_FENCE_RE.findall(stripped)
    if fence_matches:
        # Use the first fenced block that looks like JSON
        for match in fence_matches:
            candidate = match.strip()
            if candidate.startswith("[") or candidate.startswith("{"):
                return candidate

    # Strip a single wrapping fence (```json at start, ``` at end)
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()

    # If there's prose around a JSON array, try to find it
    if not stripped.startswith("["):
        bracket_start = stripped.find("[")
        if bracket_start != -1:
            # Find the matching closing bracket
            depth = 0
            for i in range(bracket_start, len(stripped)):
                if stripped[i] == "[":
                    depth += 1
                elif stripped[i] == "]":
                    depth -= 1
                    if depth == 0:
                        return stripped[bracket_start : i + 1]

    return stripped


def _validate_grid_pairs(data: list) -> Tuple[List[Dict[str, Any]], str | None]:
    """Validate that data is a list of dicts with input/output 2D int grids."""
    if not isinstance(data, list) or len(data) == 0:
        return [], "No grid pairs found in file"

    valid = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return [], f"Example {i}: not a dict"
        if "input" not in item or "output" not in item:
            return [], f"Example {i}: missing 'input' or 'output' key"

        for key in ("input", "output"):
            grid = item[key]
            if not isinstance(grid, list) or len(grid) == 0:
                return [], f"Example {i}: '{key}' must be a non-empty 2D array"
            for row in grid:
                if not isinstance(row, list):
                    return [], f"Example {i}: '{key}' rows must be arrays"
                for v in row:
                    if not isinstance(v, int) or v < 0:
                        return [], f"Example {i}: '{key}' values must be non-negative integers"

        valid.append({"input": item["input"], "output": item["output"]})

    return valid, None


async def parse_file_to_grid_pairs(
    file_content: str, filename: str
) -> Tuple[List[Dict[str, Any]], str | None]:
    """Parse file content into grid pairs, using JSON fast-path or LLM.

    Handles ChatGPT output (markdown code fences, RTF, prose wrapping).
    Returns (grid_pairs, error_message). On success error is None.
    """
    # Strip ChatGPT wrapping (code fences, RTF, prose) before any parsing
    cleaned = _strip_chatgpt_wrapping(file_content)

    # JSON fast-path: try direct parse on cleaned content (any file extension)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and len(parsed) > 0:
            pairs, err = _validate_grid_pairs(parsed)
            if err is None:
                return pairs, None
            # Validation failed — fall through to LLM
    except json.JSONDecodeError:
        pass  # Fall through to LLM

    # LLM path
    if not settings.openai_api_key:
        return [], "Could not parse as JSON — no API key configured for LLM parsing"

    try:
        from services.llm import structured_completion

        result = await structured_completion(
            system_prompt=PARSE_SYSTEM_PROMPT,
            user_content=f"Filename: {filename}\n\nContent:\n{file_content[:8000]}",
            temperature=0.3,
            max_tokens=4000,
        )

        text = _strip_chatgpt_wrapping(result)
        parsed = json.loads(text)
        return _validate_grid_pairs(parsed)

    except json.JSONDecodeError:
        return [], "LLM returned invalid JSON — try a different file format"
    except Exception as e:
        logger.warning("File parsing via LLM failed", exc_info=True)
        return [], f"Failed to parse file: {e}"
