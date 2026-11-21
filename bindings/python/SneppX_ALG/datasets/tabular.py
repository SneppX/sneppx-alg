"""Tabular datasets: dense feature/label containers plus stratification."""

from typing import List, Optional

import numpy as np


class TabularDataset:
    """An in-memory feature matrix (and optional labels).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, optional — shape ``(n_samples,)`` or ``(n_samples, k)``
    feature_names : optional list of column names
    feature_names : optional list of column names
    """

    def __init__(self, X, y=None, feature_names=None):
        self.X = np.asarray(X, dtype=np.float64)
        if self.X.ndim == 1:
            self.X = self.X[:, None]
        self.y = None if y is None else np.asarray(y)
        self.feature_names = list(feature_names) if feature_names is not None else None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.y is None:
            return x
        return x, self.y[idx]

    def head(self, n: int = 5):
        """Return the first ``n`` rows as ``(X, y)``."""
        if self.y is None:
            return self.X[:n]
        return self.X[:n], self.y[:n]

    def standardize(self):
        """Standardize features in place (zero mean, unit variance per column)."""
        mean = self.X.mean(axis=0, keepdims=True)
        std = self.X.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        self.X = (self.X - mean) / std
        return self


def class_counts(y) -> dict:
    """Return ``{label: count}`` for a label array."""
    y = np.asarray(y).ravel()
    labels, counts = np.unique(y, return_counts=True)
    return {int(l): int(c) for l, c in zip(labels, counts)}


def stratified_split(X, y, test_size: float = 0.25, seed: Optional[int] = None):
    """Class-balanced train/test split preserving per-class proportions."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for label in np.unique(y):
        cls = np.flatnonzero(y == label)
        rng.shuffle(cls)
        n_test = int(round(len(cls) * test_size))
        test_idx.extend(cls[:n_test].tolist())
        train_idx.extend(cls[n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    tr = np.asarray(train_idx)
    te = np.asarray(test_idx)
    return X[tr], X[te], y[tr], y[te]
