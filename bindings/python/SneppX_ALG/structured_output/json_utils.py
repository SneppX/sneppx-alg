"""Extract, repair and parse JSON from free-form model output (pure Python)."""

import json
import re
from typing import Any, Optional

_FENCE_RE = re.compile(r"```(?:json|json5)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def strip_code_fence(text: str) -> str:
    """Remove a surrounding Markdown code fence (```json ... ```) if present."""
    if not isinstance(text, str):
        return text
    m = re.search(r"```[a-zA-Z0-9]*\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def extract_json(text: str, prefer: Optional[str] = None) -> Optional[str]:
    """Return the first balanced JSON object/array substring, or ``None``.

    ``prefer`` may be ``"object"`` or ``"array"`` to bias the scan.
    """
    if not isinstance(text, str):
        return None
    opens = {"{": "}", "[": "]"}
    if prefer == "object":
        starts = ["{"]
    elif prefer == "array":
        starts = ["["]
    else:
        starts = ["{", "["]
    for start_ch in starts:
        idx = text.find(start_ch)
        while idx != -1:
            substr = _balanced(text, idx)
            if substr is not None:
                return substr
            idx = text.find(start_ch, idx + 1)
    return None


def _balanced(text: str, start: int) -> Optional[str]:
    stack = []
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            opening = stack.pop()
            if (opening, ch) not in (("{", "}"), ("[", "]")):
                return None
            if not stack:
                return text[start : i + 1]
    return None


def repair_json(text: str) -> str:
    """Best-effort cleanup: strip fences, drop trailing commas."""
    text = strip_code_fence(text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def loads(text: str) -> Any:
    """Parse JSON, repairing common LLM output issues on failure."""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    candidate = extract_json(text)
    if candidate is not None:
        try:
            return json.loads(candidate)
        except ValueError:
            pass
    repaired = repair_json(text if candidate is None else candidate)
    return json.loads(repaired)
