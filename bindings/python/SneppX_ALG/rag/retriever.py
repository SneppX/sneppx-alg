"""A small hybrid retriever combining BM25 and dense hashing embeddings."""

from typing import List

import numpy as np

from .bm25 import BM25
from .embeddings import EmbeddingIndex


class Retriever:
    """Index documents and retrieve the best matches for a query.

    ``mode`` is one of ``"lexical"``, ``"dense"`` or ``"hybrid"``.
    """

    def __init__(self, mode: str = "hybrid", dim: int = 256, alpha: float = 0.5):
        if mode not in {"lexical", "dense", "hybrid"}:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.alpha = float(alpha)
        self.documents: List[str] = []
        self.bm25 = BM25()
        self.embeddings = EmbeddingIndex(dim=dim)

    def add(self, documents) -> None:
        for doc in documents:
            self.documents.append(doc)
        if self.mode in ("lexical", "hybrid"):
            self.bm25.add_documents(documents)
        if self.mode in ("dense", "hybrid"):
            self.embeddings.add(documents)

    def query(self, text: str, k: int = 5):
        n = len(self.documents)
        if n == 0:
            return []
        scores = np.zeros(n, dtype=np.float64)
        if self.mode in ("lexical", "hybrid"):
            bm = self.bm25.get_scores(text)
            if bm.max() > 0:
                bm = bm / bm.max()
            w = 1.0 if self.mode == "lexical" else (1.0 - self.alpha)
            scores += w * bm
        if self.mode in ("dense", "hybrid"):
            dense = np.zeros(n, dtype=np.float64)
            for idx, score in self.embeddings.query(text, k=n):
                dense[idx] = score
            w = 1.0 if self.mode == "dense" else self.alpha
            scores += w * dense
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
