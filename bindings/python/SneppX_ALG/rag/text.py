"""Lightweight text utilities for RAG pipelines (pure Python)."""

import re
from typing import List

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "is",
    "are",
    "was",
    "were",
    "be",
    "for",
    "with",
    "as",
    "at",
    "by",
    "it",
    "this",
    "that",
}


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens."""
    return _WORD_RE.findall(text.lower())


def tokenize_no_stop(text: str) -> List[str]:
    """Word tokens with common stopwords removed."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def split_sentences(text: str) -> List[str]:
    """Naive sentence split on ``.!?`` boundaries."""
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
