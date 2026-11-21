"""Thread pool kernel bindings — thread pool, work-stealing.

Wraps C implementations in ``kernel/thread/`` with pure-Python fallback.
"""

import threading
import queue
from typing import Callable, List, Optional, Any

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_core_kernel")


class ThreadPool:
    """Simple thread pool for parallel task execution."""

    def __init__(self, n_threads: Optional[int] = None):
        self._n_threads = n_threads or (threading.cpu_count() or 4)
        self._queue: queue.Queue = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._stop = threading.Event()
        self._has_c = _HAS_C

    def start(self) -> None:
        for _ in range(self._n_threads):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                task = self._queue.get(timeout=0.5)
                if task is None:
                    break
                fn, args, kwargs = task
                fn(*args, **kwargs)
                self._queue.task_done()
            except queue.Empty:
                continue

    def submit(self, fn: Callable, *args, **kwargs) -> None:
        self._queue.put((fn, args, kwargs))

    def map(self, fn: Callable, iterable: List[Any]) -> List[Any]:
        results = []
        for item in iterable:
            self.submit(fn, item)
        return results

    def wait(self) -> None:
        self._queue.join()

    def stop(self) -> None:
        self._stop.set()
        for _ in self._workers:
            self._queue.put(None)
        for t in self._workers:
            t.join(timeout=1.0)
        self._workers.clear()

    @property
    def n_threads(self) -> int:
        return self._n_threads


class WorkStealingPool:
    """Work-stealing thread pool — idle workers steal tasks from busy workers."""

    def __init__(self, n_threads: Optional[int] = None):
        self._n_threads = n_threads or (threading.cpu_count() or 4)
        self._queues: List[queue.Queue] = [queue.Queue() for _ in range(self._n_threads)]
        self._workers: List[threading.Thread] = []
        self._stop = threading.Event()
        self._has_c = _HAS_C
        self._rr_index = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        for i in range(self._n_threads):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self._workers.append(t)

    def _worker_loop(self, worker_id: int) -> None:
        while not self._stop.is_set():
            try:
                task = self._queues[worker_id].get(timeout=0.2)
                if task is None:
                    break
                fn, args, kwargs = task
                fn(*args, **kwargs)
            except queue.Empty:
                for i in range(self._n_threads):
                    if i == worker_id:
                        continue
                    try:
                        task = self._queues[i].get_nowait()
                        if task is None:
                            break
                        fn, args, kwargs = task
                        fn(*args, **kwargs)
                        break
                    except queue.Empty:
                        continue

    def submit(self, fn: Callable, *args, **kwargs) -> None:
        with self._lock:
            idx = self._rr_index
            self._rr_index = (self._rr_index + 1) % self._n_threads
        self._queues[idx].put((fn, args, kwargs))

    def stop(self) -> None:
        self._stop.set()
        for q in self._queues:
            q.put(None)
        for t in self._workers:
            t.join(timeout=1.0)
        self._workers.clear()
