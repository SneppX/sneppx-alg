"""Timing helpers (pure Python)."""

import time
from typing import List


class Stopwatch:
    """Accumulates elapsed wall-clock time across explicit start/stop calls."""

    def __init__(self):
        self._started: List[float] = []
        self.total: float = 0.0

    def start(self) -> None:
        self._started.append(time.perf_counter())

    def stop(self) -> float:
        if not self._started:
            raise RuntimeError("stop() called without start()")
        elapsed = time.perf_counter() - self._started.pop()
        self.total += elapsed
        return elapsed

    def reset(self) -> None:
        self._started.clear()
        self.total = 0.0


class Timer:
    """A context manager that records the elapsed time in ``elapsed``."""

    def __init__(self):
        self.elapsed: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed = time.perf_counter() - self._t0
        return False
