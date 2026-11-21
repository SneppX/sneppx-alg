import time
import functools
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable


@dataclass
class ProfileEntry:
    name: str
    num_calls: int = 0
    total_time_s: float = 0.0
    min_time_s: float = float("inf")
    max_time_s: float = 0.0
    avg_time_s: float = 0.0


class Profiler:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._entries: Dict[str, ProfileEntry] = {}
        self._lock = threading.Lock()

    def record(self, name: str, elapsed_s: float):
        if not self.enabled:
            return
        with self._lock:
            if name not in self._entries:
                self._entries[name] = ProfileEntry(name=name)
            e = self._entries[name]
            e.num_calls += 1
            e.total_time_s += elapsed_s
            if elapsed_s < e.min_time_s:
                e.min_time_s = elapsed_s
            if elapsed_s > e.max_time_s:
                e.max_time_s = elapsed_s
            e.avg_time_s = e.total_time_s / e.num_calls

    def get(self, name: str) -> Optional[ProfileEntry]:
        return self._entries.get(name)

    def reset(self):
        with self._lock:
            self._entries.clear()

    def print_summary(self):
        if not self._entries:
            print("(no profiler entries)")
            return
        print(f"\n{'='*70}")
        print(
            f"{'Operation':<32} {'Calls':>8} {'Total(s)':>12} {'Avg(s)':>10} {'Min(s)':>10} {'Max(s)':>10}"
        )
        print(f"{'-'*70}")
        for name in sorted(self._entries):
            e = self._entries[name]
            print(
                f"{name:<32} {e.num_calls:>8} {e.total_time_s:>12.6f} "
                f"{e.avg_time_s:>10.6f} {e.min_time_s:>10.6f} {e.max_time_s:>10.6f}"
            )
        print(f"{'='*70}")

    def to_json(self) -> str:
        data = []
        for name in sorted(self._entries):
            e = self._entries[name]
            data.append(
                {
                    "name": name,
                    "calls": e.num_calls,
                    "total_s": e.total_time_s,
                    "avg_s": e.avg_time_s,
                    "min_s": e.min_time_s,
                    "max_s": e.max_time_s,
                }
            )
        return json.dumps({"profiler": data}, indent=2)

    def save_json(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())


# Global profiler singleton
_GLOBAL_PROFILER = Profiler()


def get_profiler() -> Profiler:
    return _GLOBAL_PROFILER


def timeit(name: Optional[str] = None):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            label = name or func.__qualname__
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t0
                _GLOBAL_PROFILER.record(label, elapsed)

        return wrapper

    return decorator


class Timer:
    def __init__(self, name: str = ""):
        self.name = name
        self._start: Optional[float] = None
        self.elapsed_s: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self._start is not None:
            self.elapsed_s = time.perf_counter() - self._start
            if self.name:
                _GLOBAL_PROFILER.record(self.name, self.elapsed_s)

    def start(self):
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is not None:
            self.elapsed_s = time.perf_counter() - self._start
        return self.elapsed_s


class MemoryTracker:
    def __init__(self):
        self._peak_mb = 0.0
        self._current_mb = 0.0

    def update(self, allocated_bytes: int):
        mb = allocated_bytes / (1024 * 1024)
        self._current_mb = mb
        if mb > self._peak_mb:
            self._peak_mb = mb

    @property
    def peak_mb(self) -> float:
        return self._peak_mb

    @property
    def current_mb(self) -> float:
        return self._current_mb

    def reset(self):
        self._peak_mb = 0.0
        self._current_mb = 0.0


class TrainProfiler:
    """Aggregates profiling for a training loop."""

    def __init__(self):
        self.profiler = Profiler()
        self.memory = MemoryTracker()
        self._step_times: List[float] = []

    def record_step(self, elapsed_s: float):
        self._step_times.append(elapsed_s)
        self.profiler.record("train_step", elapsed_s)

    def steps_per_sec(self, window: int = 100) -> float:
        recent = self._step_times[-window:]
        if not recent:
            return 0.0
        avg = sum(recent) / len(recent)
        return 1.0 / avg if avg > 0 else 0.0

    def summary(self):
        print(f"\n{'='*60}")
        print(f"  TRAINING PROFILE")
        print(f"{'='*60}")
        total = sum(self._step_times)
        print(f"  Total steps:    {len(self._step_times)}")
        print(f"  Total time:     {total:.2f}s")
        if self._step_times:
            print(
                f"  Avg step time:  {sum(self._step_times)/len(self._step_times)*1000:.2f}ms"
            )
            print(f"  Steps/sec:      {self.steps_per_sec():.1f}")
        print(f"  Peak memory:    {self.memory.peak_mb:.1f} MB")
        print(f"{'='*60}")
