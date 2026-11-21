"""Continuous batching scheduler for inference serving.

Dynamically adds/removes requests from a running batch, processing
one token at a time per request until completion.
"""

import time
import heapq
import threading
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .tensor import Tensor


class RequestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class ScheduledRequest:
    id: str
    prompt: str
    prompt_tokens: List[int]
    max_new_tokens: int
    status: RequestStatus = RequestStatus.PENDING
    generated_tokens: List[int] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    priority: int = 0
    kv_cache: Optional[Any] = None
    current_pos: int = 0
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0


@dataclass
class SchedulerConfig:
    max_batch_size: int = 32
    max_waiting_ms: float = 10.0
    max_total_tokens: int = 4096
    schedule_strategy: str = "fcfs"


@dataclass
class BatchResult:
    request_id: str
    generated_text: str
    token_ids: List[int]
    prompt_tokens: int
    completion_tokens: int
    total_time_ms: float


class ContinuousBatchScheduler:
    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self._pending: List[ScheduledRequest] = []
        self._running: Dict[str, ScheduledRequest] = {}
        self._completed: List[BatchResult] = []
        self._lock = threading.Lock()
        self._next_id = 0

    def submit(self, prompt: str, prompt_tokens: List[int],
               max_new_tokens: int = 256, priority: int = 0,
               temperature: float = 1.0, top_k: int = 0,
               top_p: float = 1.0) -> str:
        req_id = f"req_{int(time.time() * 1000)}_{self._next_id}"
        self._next_id += 1
        req = ScheduledRequest(
            id=req_id, prompt=prompt, prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens, priority=priority,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        with self._lock:
            self._pending.append(req)
        return req_id

    def get_batch(self) -> List[ScheduledRequest]:
        with self._lock:
            active = list(self._running.values())
            slots_left = self.config.max_batch_size - len(active)
            if slots_left > 0 and self._pending:
                if self.config.schedule_strategy == "fcfs":
                    batch = self._pending[:slots_left]
                    self._pending = self._pending[slots_left:]
                else:
                    self._pending.sort(key=lambda r: -r.priority)
                    batch = self._pending[:slots_left]
                    self._pending = self._pending[slots_left:]
                for req in batch:
                    req.status = RequestStatus.RUNNING
                    self._running[req.id] = req
                active.extend(batch)
        return active

    def complete(self, req_id: str, generated_tokens: List[int],
                 generated_text: str):
        with self._lock:
            req = self._running.pop(req_id, None)
            if req is None:
                return
            req.status = RequestStatus.COMPLETED
            req.end_time = time.time()
            total_ms = (req.end_time - req.start_time) * 1000
            self._completed.append(BatchResult(
                request_id=req_id, generated_text=generated_text,
                token_ids=req.prompt_tokens + generated_tokens,
                prompt_tokens=len(req.prompt_tokens),
                completion_tokens=len(generated_tokens),
                total_time_ms=total_ms,
            ))

    def cancel(self, req_id: str):
        with self._lock:
            req = self._running.pop(req_id, None)
            if req:
                req.status = RequestStatus.CANCELLED

    def get_results(self) -> List[BatchResult]:
        with self._lock:
            results = list(self._completed)
            self._completed.clear()
        return results

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "pending": len(self._pending),
                "running": len(self._running),
                "completed": len(self._completed),
                "max_batch_size": self.config.max_batch_size,
            }


def continuous_generate_loop(
    model_fn: Callable[[List[List[int]], List[int]], Tuple[List[List[int]], Any]],
    scheduler: ContinuousBatchScheduler,
    tokenizer_decode: Callable[[List[int]], str],
    max_steps: int = 1024,
):
    """Run the continuous batching generation loop."""
    for step in range(max_steps):
        batch = scheduler.get_batch()
        if not batch:
            time.sleep(0.001)
            continue

        input_ids = [req.prompt_tokens + req.generated_tokens for req in batch]
        positions = [len(ids) - 1 for ids in input_ids]

        next_tokens, kv_caches = model_fn(input_ids, positions)

        for i, req in enumerate(batch):
            next_id = next_tokens[i][0] if isinstance(next_tokens[i], list) else next_tokens[i]
            req.generated_tokens.append(next_id)

            if len(req.generated_tokens) >= req.max_new_tokens:
                text = tokenizer_decode(req.generated_tokens)
                scheduler.complete(req.id, req.generated_tokens, text)

        results = scheduler.get_results()
        if results and step % 10 == 0:
            yield results

    remaining = list(scheduler._running.values())
    for req in remaining:
        text = tokenizer_decode(req.generated_tokens)
        scheduler.complete(req.id, req.generated_tokens, text)
    yield scheduler.get_results()
