"""Learning-rate scheduling callbacks (pure Python)."""

from typing import Optional

import numpy as np

from .base import Callback


class ReduceLROnPlateau(Callback):
    """Reduce the learning rate when a monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        factor: float = 0.1,
        patience: int = 10,
        mode: str = "min",
        min_lr: float = 0.0,
        initial_lr: float = 1e-3,
        verbose: bool = False,
    ):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor = monitor
        self.factor = float(factor)
        self.patience = int(patience)
        self.mode = mode
        self.min_lr = float(min_lr)
        self.lr = float(initial_lr)
        self.best = np.inf if mode == "min" else -np.inf
        self.wait = 0
        self.num_reduce_lr = 0
        self.history = []
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        logs = logs or {}
        if self.monitor not in logs:
            return
        value = float(logs[self.monitor])
        if (self.mode == "min" and value < self.best) or (
            self.mode == "max" and value > self.best
        ):
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(self.lr * self.factor, 1e-12)
                self.lr = new_lr
                self.wait = 0
                self.num_reduce_lr += 1
        self.history.append(self.lr)


class WarmupLR(Callback):
    """Linearly warm up the learning rate over the first ``warmup_epochs``."""

    def __init__(
        self, base_lr: float, warmup_epochs: int, target_lr: Optional[float] = None
    ):
        self.base_lr = base_lr
        self.warmup_epochs = int(warmup_epochs)
        self.target_lr = target_lr if target_lr is not None else base_lr
        self.lr = base_lr

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        if self.warmup_epochs <= 0:
            self.lr = self.target_lr
            return
        frac = min(1.0, (epoch + 1) / self.warmup_epochs)
        self.lr = self.base_lr + frac * (self.target_lr - self.base_lr)
