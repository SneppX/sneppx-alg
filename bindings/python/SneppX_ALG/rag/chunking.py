"""Document chunking strategies for RAG (pure Python)."""

from typing import List, Optional

from .text import split_sentences, tokenize


def chunk_by_words(text: str, size: int, overlap: int = 0) -> List[str]:
    """Split ``text`` into word chunks of ``size`` with optional ``overlap``."""
    words = tokenize(text)
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def chunk_by_sentences(
    text: str, max_chars: int = 500, overlap_sentences: int = 0
) -> List[str]:
    """Group sentences into chunks not exceeding ``max_chars``."""
    sentences = split_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for sent in sentences:
        if current and length + len(sent) + 1 > max_chars:
            chunks.append(" ".join(current))
            if overlap_sentences > 0:
                current = current[-overlap_sentences:]
                length = sum(len(s) + 1 for s in current)
            else:
                current = []
                length = 0
        current.append(sent)
        length += len(sent) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_recursive(
    text: str, max_chars: int = 500, separators: Optional[List[str]] = None
) -> List[str]:
    """Split by progressively finer separators until chunks fit ``max_chars``."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]
    if len(text) <= max_chars or not separators:
        return [text] if text.strip() else []
    sep = separators[0]
    parts = text.split(sep)
    chunks: List[str] = []
    buffer = ""
    for part in parts:
        candidate = part if not buffer else buffer + sep + part
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            if len(part) > max_chars:
                chunks.extend(chunk_recursive(part, max_chars, separators[1:]))
                buffer = ""
            else:
                buffer = part
    if buffer:
        chunks.append(buffer)
    return [c for c in chunks if c.strip()]


def sliding_window(text: str, window: int, stride: int) -> List[str]:
    """Character-level sliding windows."""
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    out = []
    for start in range(0, max(len(text) - window, 0) + 1, stride):
        out.append(text[start : start + window])
    if not out and text:
        out.append(text)
    return out
