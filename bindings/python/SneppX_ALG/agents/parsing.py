"""Parsing of ReAct-style agent output (pure Python)."""

import re
from typing import Optional

_FINAL_RE = re.compile(r"Final Answer\s*:\s*(.*)", re.DOTALL | re.IGNORECASE)
_THOUGHT_RE = re.compile(
    r"Thought\s*:\s*(.*?)(?=\n(?:Action|Final Answer)\s*:|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_ACTION_RE = re.compile(r"Action\s*:\s*(.*?)(?=\n|$)", re.IGNORECASE)
_INPUT_RE = re.compile(
    r"Action Input\s*:\s*(.*?)(?=\n(?:Thought|Action|Final Answer|Observation)\s*:|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_OBS_RE = re.compile(r"Observation\s*:\s*(.*)", re.DOTALL | re.IGNORECASE)


def extract_thought(text: str) -> Optional[str]:
    """Return the text after ``Thought:`` up to the next section."""
    m = _THOUGHT_RE.search(text)
    return m.group(1).strip() if m else None


def extract_final(text: str) -> Optional[str]:
    """Return the text after ``Final Answer:`` or ``None``."""
    m = _FINAL_RE.search(text)
    return m.group(1).strip() if m else None


def extract_action(text: str) -> Optional[str]:
    """Return the tool name after ``Action:`` or ``None``."""
    m = _ACTION_RE.search(text)
    return m.group(1).strip() if m else None


def extract_action_input(text: str) -> Optional[str]:
    """Return the text after ``Action Input:`` or ``None``."""
    m = _INPUT_RE.search(text)
    return m.group(1).strip() if m else None


def parse(text: str) -> dict:
    """Parse a ReAct block into thought/action/action_input/final keys."""
    return {
        "thought": extract_thought(text),
        "action": extract_action(text),
        "action_input": extract_action_input(text),
        "final": extract_final(text),
    }
