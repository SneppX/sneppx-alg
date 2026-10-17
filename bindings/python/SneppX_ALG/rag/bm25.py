"""BM25 lexical ranking (pure Python / NumPy)."""

import math
from typing import List, Optional, Tuple

import numpy as np

from .text import tokenize_no_stop


class BM25:
    """Okapi BM25 with the standard Lucene-style IDF."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.corpus: List[List[str]] = []
        self.doc_len: List[int] = []
        self.doc_freqs: List[dict] = []
        self.df: dict = {}
        self.idf: dict = {}
        self.avgdl: float = 0.0
        self._dirty = True

    def add_document(self, document: str) -> None:
        tokens = tokenize_no_stop(document)
        self.corpus.append(tokens)
        self.doc_len.append(len(tokens))
        freqs: dict = {}
        for t in tokens:
            freqs[t] = freqs.get(t, 0) + 1
        self.doc_freqs.append(freqs)
        for t in freqs:
            self.df[t] = self.df.get(t, 0) + 1
        self._dirty = True

    def add_documents(self, documents) -> None:
        for doc in documents:
            self.add_document(doc)

    def _build(self) -> None:
        n = len(self.corpus)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        self._dirty = False

    def get_scores(self, query: str) -> np.ndarray:
        if self._dirty:
            self._build()
        q_tokens = tokenize_no_stop(query)
        scores = np.zeros(len(self.corpus), dtype=np.float64)
        for i, freqs in enumerate(self.doc_freqs):
            dl = self.doc_len[i]
            norm = 1.0 - self.b + self.b * (dl / self.avgdl) if self.avgdl else 0.0
            s = 0.0
            for term in q_tokens:
                if term not in freqs or term not in self.idf:
                    continue
                f = freqs[term]
                s += self.idf[term] * (f * (self.k1 + 1.0)) / (f + self.k1 * norm)
            scores[i] = s
        return scores

    def top_k(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        scores = self.get_scores(query)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
