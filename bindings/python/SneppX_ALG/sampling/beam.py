"""Beam search over a step-wise log-probability model (pure NumPy)."""

from typing import Callable, List, Optional, Tuple

import numpy as np


def beam_search(
    step_logprobs: Callable[[List[int]], np.ndarray],
    bos: int,
    eos: Optional[int] = None,
    beam_width: int = 3,
    max_len: int = 16,
    length_penalty: float = 0.0,
) -> Tuple[List[int], float]:
    """Generic beam search.

    ``step_logprobs(prefix)`` returns a vector of log-probabilities for the
    next token given the token ``prefix`` (which starts as ``[bos]``).
    Returns the highest-scoring sequence (excluding ``bos``/``eos``) and its
    length-normalized score.
    """

    def norm_score(logprob: float, length: int) -> float:
        if length_penalty == 0.0 or length == 0:
            return logprob
        return logprob / (length**length_penalty)

    beams: List[Tuple[List[int], float]] = [([bos], 0.0)]
    finished: List[Tuple[List[int], float]] = []
    for _ in range(max_len):
        candidates: List[Tuple[List[int], float]] = []
        for tokens, score in beams:
            lp = np.asarray(step_logprobs(tokens), dtype=np.float64)
            top = np.argsort(-lp)[:beam_width]
            for tok in top:
                tok = int(tok)
                cand = tokens + [tok]
                new_score = score + float(lp[tok])
                if eos is not None and tok == eos:
                    finished.append((cand, new_score))
                else:
                    candidates.append((cand, new_score))
        if not candidates:
            break
        candidates.sort(key=lambda t: norm_score(t[1], len(t[0]) - 1), reverse=True)
        beams = candidates[:beam_width]

    pool = finished if finished else beams
    pool.sort(key=lambda t: norm_score(t[1], len(t[0]) - 1), reverse=True)
    best_tokens, best_score = pool[0]

    out = best_tokens[1:]
    if eos is not None and eos in out:
        out = out[: out.index(eos)]
    return out, norm_score(best_score, len(out))
