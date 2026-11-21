"""Early stopping and weight-averaging callbacks (pure Python)."""

from typing import Optional

import numpy as np


class EarlyStopping:
    """Track a metric and signal when training should stop.

    Parameters
    ----------
    patience : int — number of non-improving epochs to tolerate
    min_delta : float — minimum change to count as an improvement
    mode : {"min", "max"} — whether lower or higher is better
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best: bool = True,
    ):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.restore_best = restore_best
        self.best = np.inf if mode == "min" else -np.inf
        self.best_epoch = -1
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch: Optional[int] = None
        self._epoch = 0

    def _improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float, weights=None) -> bool:
        """Update with a new metric value; return True if training should stop."""
        improved = self._improved(value)
        if improved:
            self.best = float(value)
            self.best_epoch = self._epoch
            self.best_weights = (
                None if weights is None else np.array(weights, copy=True)
            )
            self.wait = 0
        else:
            self.wait += 1
        self._epoch += 1
        return self.wait >= self.patience

    @property
    def stopped_epoch_(self) -> Optional[int]:
        return self._epoch - self.wait if self.wait >= self.patience else None


class WeightAveraging:
    """Maintain an exponential moving average of parameters."""

    def __init__(self, decay: float = 0.99):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = float(decay)
        self.average = None

    def update(self, params):
        params = np.asarray(params, dtype=np.float64)
        if self.average is None:
            self.average = params.copy()
        else:
            self.average = self.decay * self.average + (1.0 - self.decay) * params
        return self.average
