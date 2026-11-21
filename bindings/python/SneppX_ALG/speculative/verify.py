"""Acceptance rules for speculative decoding (pure Python / NumPy)."""

from typing import List, Optional

import numpy as np


def acceptance_probability(p_draft, p_target, token: int) -> float:
    """min(1, p_target[token] / p_draft[token]); 1.0 if the draft is impossible."""
    p_d = float(p_draft[token])
    p_t = float(p_target[token])
    if p_d <= 0:
        return 1.0
    return float(min(1.0, p_t / p_d))


def verify_greedy(draft_tokens: List[int], target_tokens: List[int]) -> int:
    """Return the number of leading draft tokens matching the target's greedy output."""
    n = 0
    for d, t in zip(draft_tokens, target_tokens):
        if d != t:
            break
        n += 1
    return n


def speculative_accept(
    draft_tokens: List[int],
    draft_probs: List[np.ndarray],
    target_probs: List[np.ndarray],
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Rejection-sampling acceptance; returns the accepted prefix.

    Accepts each draft token ``x`` with probability ``min(1, p_t(x)/p_d(x))``,
    stopping at the first rejection. Identical in distribution to sampling from
    the target when the draft is used only to propose.
    """
    rng = rng or np.random.default_rng()
    accepted: List[int] = []
    for token, pd, pt in zip(draft_tokens, draft_probs, target_probs):
        p_accept = acceptance_probability(pd, pt, token)
        if rng.random() <= p_accept:
            accepted.append(token)
        else:
            break
    return accepted


def residual_distribution(p_draft, p_target) -> np.ndarray:
    """``max(0, p_target - p_draft)`` renormalized (for the correction sample)."""
    pd = np.asarray(p_draft, dtype=np.float64)
    pt = np.asarray(p_target, dtype=np.float64)
    resid = np.maximum(0.0, pt - pd)
    total = resid.sum()
    if total <= 0:
        return pt / pt.sum() if pt.sum() > 0 else pt
    return resid / total
