"""Logging callbacks — in-memory history and CSV logging (pure Python)."""

import csv
import os
from typing import Dict, List, Optional, Union

from .base import Callback


class History(Callback):
    """Record metric values per epoch into ``self.history`` (dict of lists)."""

    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self.epochs: List[int] = []

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        self.epochs.append(epoch)
        for key, value in (logs or {}).items():
            self.history.setdefault(key, []).append(value)


class CSVLogger(Callback):
    """Append metrics to a CSV file, one row per epoch."""

    def __init__(self, filepath: Union[str, os.PathLike], separator: str = ","):
        self.filepath = str(filepath)
        self.separator = separator
        self._header: Optional[List[str]] = None

    def _write_header(self, keys: List[str]):
        self._header = ["epoch"] + keys
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f, delimiter=self.separator)
            writer.writerow(self._header)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        logs = logs or {}
        if self._header is None:
            self._write_header(list(logs.keys()))
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f, delimiter=self.separator)
            writer.writerow([epoch] + [logs.get(k, "") for k in self._header[1:]])


class LambdaCallback(Callback):
    """Wrap plain functions as a callback for quick prototyping."""

    def __init__(
        self,
        on_epoch_end=None,
        on_train_begin=None,
        on_train_end=None,
    ):
        self._on_epoch_end = on_epoch_end
        self._on_train_begin = on_train_begin
        self._on_train_end = on_train_end

    def on_train_begin(self, logs=None):
        if self._on_train_begin:
            self._on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        if self._on_epoch_end:
            self._on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        if self._on_train_end:
            self._on_train_end(logs)
