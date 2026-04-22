from __future__ import annotations

import io
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

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


def _parse_xlsx_bytes(raw: bytes) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Extract grid pairs from an xlsx workbook.

    Expected layout (first sheet, first row optional header):
      A: JSON string of input grid  (e.g. `[[1,0],[0,1]]`)
      B: JSON string of output grid

    Also accepts a "grid of cells" layout where a single row of cells is
    itself a 1xN grid — useful for small demos.
    """
    try:
        import openpyxl  # type: ignore
    except ImportError:
        return [], "xlsx support not installed (openpyxl missing)"

    try:
        wb = openpyxl.load_workbook(io.BytesIO(raw), data_only=True, read_only=True)
    except Exception as e:
        return [], f"Could not open xlsx: {e}"

    ws = wb.active
    if ws is None:
        return [], "Workbook has no active sheet"

    pairs: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
        if not row:
            continue
        a = row[0]
        b = row[1] if len(row) > 1 else None
        if a is None or b is None:
            continue
        # Skip a header row like ("input", "output").
        if row_idx == 0 and isinstance(a, str) and isinstance(b, str):
            if a.strip().lower() in {"input", "inputs"} and b.strip().lower() in {
                "output",
                "outputs",
            }:
                continue

        def _coerce(cell: Any) -> Optional[List[List[int]]]:
            if isinstance(cell, str):
                try:
                    parsed = json.loads(cell)
                except json.JSONDecodeError:
                    return None
            else:
                parsed = cell
            if not isinstance(parsed, list):
                return None
            # Flat list of ints -> treat as 1xN grid.
            if all(isinstance(v, (int, float)) for v in parsed):
                return [[int(v) for v in parsed]]
            grid: List[List[int]] = []
            for r in parsed:
                if not isinstance(r, list):
                    return None
                try:
                    grid.append([int(v) for v in r])
                except (TypeError, ValueError):
                    return None
            return grid

        inp = _coerce(a)
        out = _coerce(b)
        if inp is None or out is None:
            continue
        pairs.append({"input": inp, "output": out})

    if not pairs:
        return [], "No valid input/output pairs found in xlsx"
    return pairs, None


async def parse_file_to_grid_pairs(
    file_content: str, filename: str, raw_bytes: Optional[bytes] = None
) -> Tuple[List[Dict[str, Any]], str | None]:
    """Parse file content into grid pairs, using JSON fast-path or LLM.

    Handles ChatGPT output (markdown code fences, RTF, prose wrapping).
    Returns (grid_pairs, error_message). On success error is None.
    """
    lower_name = (filename or "").lower()
    if raw_bytes is not None and (
        lower_name.endswith(".xlsx")
        or lower_name.endswith(".xlsm")
        or lower_name.endswith(".xltx")
    ):
        return _parse_xlsx_bytes(raw_bytes)

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
