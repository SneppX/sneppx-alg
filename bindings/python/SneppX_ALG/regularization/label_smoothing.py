"""Label smoothing (pure NumPy)."""

import numpy as np


def smooth_one_hot(y_onehot, eps: float = 0.1) -> np.ndarray:
    """Smooth a one-hot label matrix.

    ``target = (1 - eps) * y + eps / K`` where ``K`` is the number of classes.
    """
    y = np.asarray(y_onehot, dtype=np.float64)
    k = y.shape[-1]
    return y * (1.0 - eps) + eps / k


def smooth_labels(y, num_classes: int, eps: float = 0.1) -> np.ndarray:
    """One-hot encode integer labels then apply label smoothing."""
    y = np.asarray(y).ravel().astype(np.int64)
    onehot = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    valid = (y >= 0) & (y < num_classes)
    onehot[np.arange(y.shape[0])[valid], y[valid]] = 1.0
    return smooth_labels_onehot(onehot, eps)


def smooth_labels_onehot(onehot, eps: float = 0.1) -> np.ndarray:
    return smooth_one_hot(onehot, eps)


def confidence_penalty(probs, eps: float = 1e-3) -> float:
    """Confidence penalty (Pereyra et al.): ``-sum p log p`` (entropy, mean)."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0)
    return float(-np.mean(np.sum(probs * np.log(probs), axis=-1)))
