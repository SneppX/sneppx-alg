"""Simple text progress bar (no external dependencies)."""

import sys
import time
from typing import Optional, TextIO

from .base import Callback


class ProgressBar(Callback):
    """A minimal epoch progress bar drawn to ``stream`` (default stderr)."""

    def __init__(
        self,
        total_epochs: int,
        width: int = 30,
        stream: Optional[TextIO] = None,
        monitor: str = "loss",
    ):
        self.total_epochs = int(total_epochs)
        self.width = int(width)
        self.stream = stream if stream is not None else sys.stderr
        self.monitor = monitor
        self.start_time = 0.0
        self._last_len = 0

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def _render(self, epoch: int, logs: Optional[dict]):
        frac = (epoch + 1) / max(self.total_epochs, 1)
        fill = int(self.width * min(frac, 1.0))
        bar = "#" * fill + "-" * (self.width - fill)
        elapsed = time.time() - self.start_time
        metric = ""
        if logs and self.monitor in logs:
            metric = f" {self.monitor}={logs[self.monitor]:.4f}"
        line = f"\r[{bar}] {epoch + 1}/{self.total_epochs}{metric} {elapsed:.1f}s"
        self.stream.write(line)
        self.stream.flush()
        self._last_len = len(line)

    def on_epoch_end(self, epoch: int, logs=None):
        self._render(epoch, logs)

    def on_train_end(self, logs=None):
        self.stream.write("\n")
        self.stream.flush()
