"""Constrained-decoding helpers: logit masking and finite-choice trie (pure Python)."""

from typing import Dict, Iterable, List, Optional

import numpy as np

NEG_INF = -1e9


def mask_logits(
    logits, allowed_token_ids: Iterable[int], neg_value: float = NEG_INF
) -> np.ndarray:
    """Return a copy of ``logits`` with disallowed token ids set to ``neg_value``."""
    out = np.array(logits, dtype=np.float64, copy=True)
    allowed = set(int(i) for i in allowed_token_ids)
    if out.ndim == 1:
        for i in range(out.shape[0]):
            if i not in allowed:
                out[i] = neg_value
    else:
        raise ValueError("mask_logits expects a 1-D logit vector")
    return out


def top_token(logits, allowed_token_ids: Optional[Iterable[int]] = None) -> int:
    """Arg-max over allowed tokens (all tokens if unspecified)."""
    if allowed_token_ids is not None:
        logits = mask_logits(logits, allowed_token_ids)
    return int(np.argmax(logits))


class Constrainer:
    """Base class for a character-level decoding constraint."""

    def allowed(self) -> set:
        """Set of characters that may be emitted next (empty = complete)."""
        raise NotImplementedError

    def advance(self, ch: str) -> None:
        raise NotImplementedError

    def is_complete(self) -> bool:
        raise NotImplementedError


class ChoiceConstrainer(Constrainer):
    """Constrain output to one of a finite set of candidate strings.

    Keeps the set of candidates consistent with the emitted prefix and reports
    the allowed next characters; the trie is implicit in the candidate set.
    """

    def __init__(self, candidates: Iterable[str], allow_eos: bool = True):
        self.candidates = list(dict.fromkeys(candidates))
        if not self.candidates:
            raise ValueError("candidates must be non-empty")
        self.prefix = ""
        self.allow_eos = allow_eos

    def _matching(self) -> List[str]:
        return [c for c in self.candidates if c.startswith(self.prefix)]

    def allowed(self) -> set:
        chars = set()
        for c in self._matching():
            if len(c) > len(self.prefix):
                chars.add(c[len(self.prefix)])
        return chars

    def advance(self, ch: str) -> None:
        if ch not in self.allowed():
            raise ValueError(f"character {ch!r} not allowed after {self.prefix!r}")
        self.prefix += ch

    def is_complete(self) -> bool:
        return self.prefix in self.candidates
