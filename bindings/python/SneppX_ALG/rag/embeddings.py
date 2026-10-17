"""Hashing-based embeddings and a cosine index (pure Python / NumPy).

The default embedding is a deterministic feature-hashing bag-of-words vector,
so retrieval works without a downloaded model or network access.
"""

import zlib
from typing import List, Tuple

import numpy as np

from .text import tokenize_no_stop


def hashing_embed(text: str, dim: int = 256) -> np.ndarray:
    """Hashing-trick bag-of-words embedding (L2-normalized)."""
    vec = np.zeros(dim, dtype=np.float64)
    tokens = tokenize_no_stop(text)
    for tok in tokens:
        h = zlib.crc32(tok.encode("utf-8"))
        idx = h % dim
        sign = 1.0 if (h >> 16) & 1 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class EmbeddingIndex:
    """In-memory dense index using hashing embeddings + cosine similarity."""

    def __init__(self, dim: int = 256):
        self.dim = int(dim)
        self.documents: List[str] = []
        self._matrix = None

    def add(self, documents) -> None:
        for doc in documents:
            self.documents.append(doc)
            vec = hashing_embed(doc, self.dim).reshape(1, -1)
            self._matrix = (
                vec if self._matrix is None else np.vstack([self._matrix, vec])
            )

    def query(self, text: str, k: int = 5):
        if self._matrix is None or not self.documents:
            return []
        q = hashing_embed(text, self.dim)
        sims = self._matrix @ q
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order if sims[i] > 0]
