"""Model checkpointing callback (pure Python / NumPy)."""

from typing import Callable, Dict, Optional

import numpy as np

from .base import Callback


class ModelCheckpoint(Callback):
    """Keep track of the best model parameters by a monitored metric.

    Parameters
    ----------
    monitor : str — key in ``logs`` to watch
    mode : {"min", "max"}
    params_fn : callable returning a ``{name: ndarray}`` snapshot
    save_best_only : bool
    filepath : optional ``.npz`` path to persist the best snapshot
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        params_fn: Optional[Callable[[], Dict[str, np.ndarray]]] = None,
        save_best_only: bool = True,
        filepath: Optional[str] = None,
    ):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor = monitor
        self.mode = mode
        self.params_fn = params_fn
        self.save_best_only = save_best_only
        self.filepath = filepath
        self.best = np.inf if mode == "min" else -np.inf
        self.best_epoch = -1
        self.best_params: Optional[Dict[str, np.ndarray]] = None
        self.saved = False

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best
        return value > self.best

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        logs = logs or {}
        if self.monitor not in logs:
            return
        value = float(logs[self.monitor])
        improved = self._is_improvement(value)
        if improved or not self.save_best_only:
            self.saved = True
            if self.params_fn is not None:
                snapshot = {
                    k: np.array(v, copy=True) for k, v in self.params_fn().items()
                }
                self.best_params = snapshot
                if self.filepath:
                    np.savez(self.filepath, **snapshot)
        if improved:
            self.best = value
            self.best_epoch = epoch

    def restore(self) -> Optional[Dict[str, np.ndarray]]:
        return self.best_params
