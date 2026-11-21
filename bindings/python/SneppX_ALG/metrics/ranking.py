"""Ranking metrics (pure NumPy).

Precision/recall@k, MAP@k, MRR, NDCG and hit-rate for ranked recommendation
and retrieval lists. ``relevant`` is a binary sequence in ranked order.
"""

from typing import Sequence

import numpy as np


def _rel(relevant) -> np.ndarray:
    return np.asarray(relevant, dtype=np.float64).ravel()


def precision_at_k(relevant, k: int) -> float:
    r = _rel(relevant)[:k]
    return float(r.mean()) if r.size else 0.0


def recall_at_k(relevant, k: int, total_relevant=None) -> float:
    r = _rel(relevant)
    total = int(total_relevant) if total_relevant is not None else int(r.sum())
    if total == 0:
        return 0.0
    return float(r[:k].sum() / total)


def hit_rate_at_k(relevant, k: int) -> float:
    return 1.0 if _rel(relevant)[:k].sum() > 0 else 0.0


def average_precision_at_k(relevant, k: int) -> float:
    r = _rel(relevant)[:k]
    if r.sum() == 0:
        return 0.0
    hits = np.cumsum(r)
    positions = np.arange(1, r.size + 1)
    precision = hits / positions
    return float((precision * r).sum() / r.sum())


def mean_average_precision_at_k(relevant_lists: Sequence, k: int) -> float:
    vals = [average_precision_at_k(r, k) for r in relevant_lists]
    return float(np.mean(vals)) if vals else 0.0


def reciprocal_rank(relevant) -> float:
    r = _rel(relevant)
    hits = np.flatnonzero(r > 0)
    return float(1.0 / (hits[0] + 1)) if hits.size else 0.0


def mean_reciprocal_rank(relevant_lists: Sequence) -> float:
    vals = [reciprocal_rank(r) for r in relevant_lists]
    return float(np.mean(vals)) if vals else 0.0


def dcg_at_k(relevances, k: int) -> float:
    r = _rel(relevances)[:k]
    if r.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, r.size + 2))
    return float(((2**r - 1) / discounts).sum())


def ndcg_at_k(relevances, k: int) -> float:
    r = _rel(relevances)
    ideal = np.sort(r)[::-1]
    dcg = dcg_at_k(r, k)
    idcg = dcg_at_k(ideal, k)
    return float(dcg / idcg) if idcg > 0 else 0.0


def mean_ndcg_at_k(relevant_lists: Sequence, k: int) -> float:
    vals = [ndcg_at_k(r, k) for r in relevant_lists]
    return float(np.mean(vals)) if vals else 0.0
