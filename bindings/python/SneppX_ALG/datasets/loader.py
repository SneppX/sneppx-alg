"""A minimal NumPy-backed data loader for minibatch iteration."""

from typing import Optional

import numpy as np

from .utils import batch_indices


class ArrayDataLoader:
    """Iterate over ``(X, y)`` in minibatches.

    Parameters
    ----------
    X : array-like of shape ``(n, ...)``
    y : optional array-like of shape ``(n, ...)``
    batch_size : int
    shuffle : bool — reshuffle index order once per epoch
    drop_last : bool — drop a final partial batch
    seed : optional RNG seed
    """

    def __init__(
        self,
        X,
        y=None,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        self.X = np.asarray(X)
        self.y = None if y is None else np.asarray(y)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = seed
        self._epoch = 0
        if self.X.shape[0] != (
            self.y.shape[0] if self.y is not None else self.X.shape[0]
        ):
            raise ValueError("X and y must have the same length")

    def __len__(self) -> int:
        n = self.X.shape[0]
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = self.X.shape[0]
        seed = None if self.seed is None else self.seed + self._epoch
        self._epoch += 1
        for idx in batch_indices(
            n, self.batch_size, self.drop_last, self.shuffle, seed
        ):
            xb = self.X[idx]
            if self.y is None:
                yield xb
            else:
                yield xb, self.y[idx]
