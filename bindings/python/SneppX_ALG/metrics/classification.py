"""Classification metrics (pure NumPy).

Binary and multiclass metrics matching the semantics of the corresponding
scikit-learn functions: ``accuracy_score``, ``confusion_matrix``,
``precision_recall_fscore_support``, ``roc_auc_score`` and friends.
"""

from typing import Optional, Tuple

import numpy as np


def _flat_int(x) -> np.ndarray:
    return np.asarray(x).ravel().astype(np.int64)


def confusion_matrix(y_true, y_pred, num_classes: Optional[int] = None) -> np.ndarray:
    """Return the ``(C, C)`` confusion matrix ``cm[true, pred]``."""
    yt = _flat_int(y_true)
    yp = _flat_int(y_pred)
    if num_classes is None:
        hi = int(max(yt.max(initial=0), yp.max(initial=0)))
        num_classes = hi + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (yt, yp), 1)
    return cm


def accuracy_score(y_true, y_pred) -> float:
    """Fraction of exactly correct predictions."""
    yt = _flat_int(y_true)
    yp = _flat_int(y_pred)
    return float(np.mean(yt == yp)) if yt.size else 0.0


def top_k_accuracy_score(y_true, y_score, k: int = 5) -> float:
    """Fraction of samples whose true label is in the top-``k`` scores."""
    yt = _flat_int(y_true)
    ys = np.asarray(y_score, dtype=np.float64)
    if ys.ndim == 1:
        ys = np.stack([1.0 - ys, ys], axis=1)
    k = min(k, ys.shape[1])
    topk = np.argsort(-ys, axis=1)[:, :k]
    hits = (topk == yt[:, None]).any(axis=1)
    return float(np.mean(hits)) if hits.size else 0.0


def precision_recall_fscore_support(
    y_true, y_pred, num_classes: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-class precision, recall, F1 and support."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    support = cm.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    return precision, recall, f1, support


def _aggregate(
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    support: np.ndarray,
    average,
) -> Tuple[float, float, float]:
    if average is None:
        raise ValueError("average=None returns arrays, use the _support helper")
    if average == "macro":
        return float(precision.mean()), float(recall.mean()), float(f1.mean())
    if average == "weighted":
        w = support.astype(np.float64)
        tot = w.sum()
        if tot == 0:
            return 0.0, 0.0, 0.0
        return (
            float((precision * w).sum() / tot),
            float((recall * w).sum() / tot),
            float((f1 * w).sum() / tot),
        )
    if average == "binary":
        return float(precision[-1]), float(recall[-1]), float(f1[-1])
    raise ValueError(f"unsupported average: {average!r}")


def precision_score(
    y_true, y_pred, average: str = "binary", pos_label: int = 1
) -> float:
    yt = _flat_int(y_true)
    yp = _flat_int(y_pred)
    if average == "micro":
        tp = int(((yt == pos_label) & (yp == pos_label)).sum())
        fp = int(((yt != pos_label) & (yp == pos_label)).sum())
        fn = int(((yt == pos_label) & (yp != pos_label)).sum())
        return float(tp / (tp + fp)) if (tp + fp) else 0.0
    p, _, _, s = precision_recall_fscore_support(yt, yp)
    return _aggregate(p, _, _, s, average)[0]


def recall_score(y_true, y_pred, average: str = "binary", pos_label: int = 1) -> float:
    yt = _flat_int(y_true)
    yp = _flat_int(y_pred)
    if average == "micro":
        tp = int(((yt == pos_label) & (yp == pos_label)).sum())
        false_neg = int(((yt == pos_label) & (yp != pos_label)).sum())
        return float(tp / (tp + false_neg)) if (tp + false_neg) else 0.0
    _, r, _, s = precision_recall_fscore_support(yt, yp)
    return _aggregate(_, r, _, s, average)[1]


def f1_score(y_true, y_pred, average: str = "binary", pos_label: int = 1) -> float:
    yt = _flat_int(y_true)
    yp = _flat_int(y_pred)
    if average == "micro":
        p = precision_score(yt, yp, "micro", pos_label)
        r = recall_score(yt, yp, "micro", pos_label)
        return float(2 * p * r / (p + r)) if (p + r) else 0.0
    _, _, f, s = precision_recall_fscore_support(yt, yp)
    return _aggregate(_, _, f, s, average)[2]


def roc_auc_score(y_true, y_score) -> float:
    """Area under the ROC curve. Binary labels; scores are probabilities.

    Uses the rank (Mann-Whitney U) formulation, which handles ties.
    """
    yt = _flat_int(y_true)
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    pos = yt == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(ys, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, ys.size + 1)
    # average ranks for ties
    sorted_ys = ys[order]
    i = 0
    while i < ys.size:
        j = i
        while j + 1 < ys.size and sorted_ys[j + 1] == sorted_ys[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    rank_sum_pos = float(ranks[pos].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision_score(y_true, y_score) -> float:
    """Area under the precision-recall curve (step interpolation)."""
    yt = _flat_int(y_true)
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    order = np.argsort(-ys, kind="mergesort")
    yt = yt[order]
    tp = np.cumsum(yt == 1)
    fp = np.cumsum(yt != 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((yt == 1).sum()), 1)
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


def matthews_corrcoef(y_true, y_pred) -> float:
    """Matthews correlation coefficient (binary), in ``[-1, 1]``."""
    cm = confusion_matrix(y_true, y_pred, num_classes=2)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def balanced_accuracy_score(y_true, y_pred) -> float:
    """Mean per-class recall (robust to class imbalance)."""
    cm = confusion_matrix(y_true, y_pred)
    row = cm.sum(axis=1).astype(np.float64)
    diag = np.diag(cm).astype(np.float64)
    present = row > 0
    if not present.any():
        return 0.0
    return float((diag[present] / row[present]).mean())


def cohen_kappa_score(y_true, y_pred) -> float:
    """Cohen's kappa: agreement corrected for chance."""
    cm = confusion_matrix(y_true, y_pred).astype(np.float64)
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.diag(cm).sum() / n
    pe = float((cm.sum(axis=0) * cm.sum(axis=1)).sum()) / (n * n)
    if abs(1.0 - pe) < 1e-12:
        return 0.0
    return float((po - pe) / (1.0 - pe))


__all__ = [
    "confusion_matrix",
    "accuracy_score",
    "top_k_accuracy_score",
    "precision_recall_fscore_support",
    "precision_score",
    "recall_score",
    "f1_score",
    "roc_auc_score",
    "average_precision_score",
    "matthews_corrcoef",
    "balanced_accuracy_score",
    "cohen_kappa_score",
]
