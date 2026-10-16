"""Dataset utilities — splitting, batching, encoders and normalization.

Pure-NumPy helpers with no external dependencies. Works on arrays, lists or
anything convertible to ``np.ndarray``.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np


def _as_arrays(*arrays) -> List[np.ndarray]:
    out = []
    for a in arrays:
        arr = np.asarray(a)
        out.append(arr)
    return out


def train_test_split(
    *arrays,
    test_size: float = 0.25,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Split one or more parallel arrays into train/test halves.

    Returns ``[train_a, test_a, train_b, test_b, ...]`` in input order.
    """
    arrs = _as_arrays(*arrays)
    if not arrs:
        return []
    n = arrs[0].shape[0]
    for a in arrs:
        if a.shape[0] != n:
            raise ValueError("all arrays must share the same first dimension")
    if test_size < 1:
        n_test = int(round(n * test_size))
    else:
        n_test = int(test_size)
    n_test = min(max(n_test, 0), n)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    result: List[np.ndarray] = []
    for a in arrs:
        result.append(a[train_idx])
        result.append(a[test_idx])
    return result


def k_fold_indices(
    n: int, n_splits: int = 5, shuffle: bool = True, seed: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(train_idx, val_idx)`` pairs for k-fold cross validation."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    result = []
    for i in range(n_splits):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        result.append((train, val))
    return result


def batch_indices(
    n: int,
    batch_size: int,
    drop_last: bool = False,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Yield index arrays for minibatch iteration."""
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    batches = [idx[i : i + batch_size] for i in range(0, n, batch_size)]
    if drop_last and batches and batches[-1].size < batch_size:
        batches = batches[:-1]
    return batches


def one_hot(labels, num_classes: Optional[int] = None) -> np.ndarray:
    """One-hot encode integer labels -> ``(N, num_classes)`` float32."""
    labels = np.asarray(labels).ravel().astype(np.int64)
    if num_classes is None:
        num_classes = int(labels.max()) + 1 if labels.size else 0
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    valid = (labels >= 0) & (labels < num_classes)
    out[np.arange(labels.shape[0])[valid], labels[valid]] = 1.0
    return out


def normalize(
    x, mean=None, std: Optional[np.ndarray] = None, axis: int = 0, eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize ``x`` per ``axis``. Returns ``(normalized, mean, std)``."""
    x = np.asarray(x, dtype=np.float64)
    if mean is None:
        mean = x.mean(axis=axis, keepdims=True)
    else:
        mean = np.asarray(mean, dtype=np.float64)
    if std is None:
        std = x.std(axis=axis, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
    else:
        std = np.asarray(std, dtype=np.float64)
    return (x - mean) / std, mean, std


def one_hot_decode(onehot) -> np.ndarray:
    """Inverse of ``one_hot``: argmax over the last axis."""
    return np.asarray(onehot).argmax(axis=-1)
