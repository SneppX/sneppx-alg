"""Draft models for speculative decoding (pure Python)."""

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


class NGramDrafter:
    """Proposes continuations from an n-gram table built over a corpus."""

    def __init__(self, n: int = 3, max_draft: int = 4):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.max_draft = max_draft
        self.table: Dict[Tuple[int, ...], List[int]] = defaultdict(list)

    def fit(self, corpus: Iterable[Iterable[int]]) -> "NGramDrafter":
        for seq in corpus:
            tokens = list(seq)
            for i in range(len(tokens) - self.n):
                key = tuple(tokens[i : i + self.n])
                self.table[key].append(tokens[i + self.n])
        return self

    def _next(self, prefix: List[int]) -> int:
        if len(prefix) < self.n:
            return None
        key = tuple(prefix[-self.n :])
        candidates = self.table.get(key)
        if not candidates:
            return None
        counts: Dict[int, int] = {}
        for tok in candidates:
            counts[tok] = counts.get(tok, 0) + 1
        return max(counts, key=counts.get)

    def draft(self, prefix: List[int], k: int = None) -> List[int]:
        k = self.max_draft if k is None else k
        out: List[int] = []
        working = list(prefix)
        for _ in range(k):
            nxt = self._next(working)
            if nxt is None:
                break
            out.append(nxt)
            working.append(nxt)
        return out


class PromptLookupDrafter:
    """Copies the continuation of an earlier occurrence of the last n-gram."""

    def __init__(self, n: int = 2, max_draft: int = 4):
        self.n = n
        self.max_draft = max_draft

    def draft(self, prefix: List[int], k: int = None) -> List[int]:
        k = self.max_draft if k is None else k
        if len(prefix) < self.n:
            return []
        key = prefix[-self.n :]
        for i in range(len(prefix) - self.n - 1, -1, -1):
            if prefix[i : i + self.n] == key:
                cont = prefix[i + self.n : i + self.n + k]
                return list(cont)
        return []
