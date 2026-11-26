"""Tests for continuous batching scheduler."""

import time
import threading
import numpy as np
import pytest
from SneppX_ALG.interface_bindings.continuous_batching import (
    ContinuousBatchScheduler,
    SchedulerConfig,
    ScheduledRequest,
    RequestStatus,
)


def test_scheduler_create():
    s = ContinuousBatchScheduler()
    assert s.config.max_batch_size == 32
    print("  test_scheduler_create PASS")


def test_submit_and_stats():
    s = ContinuousBatchScheduler()
    rid = s.submit("hello", [1, 2, 3], max_new_tokens=10)
    assert rid.startswith("req_")
    stats = s.get_stats()
    assert stats["pending"] == 1
    print("  test_submit_and_stats PASS")


def test_get_batch_fcfs():
    s = ContinuousBatchScheduler()
    s.submit("a", [1], max_new_tokens=5, priority=0)
    s.submit("b", [2], max_new_tokens=5, priority=10)
    batch = s.get_batch()
    assert len(batch) == 2
    assert batch[0].priority == 0
    assert batch[1].priority == 10
    print("  test_get_batch_fcfs PASS")


def test_complete_and_results():
    s = ContinuousBatchScheduler()
    rid = s.submit("hi", [1, 2], max_new_tokens=3)
    s.get_batch()
    s.complete(rid, [3, 4, 5], " world")
    results = s.get_results()
    assert len(results) == 1
    assert results[0].request_id == rid
    assert results[0].completion_tokens == 3
    assert results[0].prompt_tokens == 2
    print("  test_complete_and_results PASS")


def test_cancel():
    s = ContinuousBatchScheduler()
    rid = s.submit("hi", [1, 2])
    s.get_batch()
    s.cancel(rid)
    with s._lock:
        assert rid not in s._running
    print("  test_cancel PASS")


def test_max_batch_size():
    s = ContinuousBatchScheduler(SchedulerConfig(max_batch_size=2))
    for i in range(5):
        s.submit(f"p{i}", [i], max_new_tokens=2)
    batch = s.get_batch()
    assert len(batch) == 2
    print("  test_max_batch_size PASS")


def test_empty_batch():
    s = ContinuousBatchScheduler()
    batch = s.get_batch()
    assert batch == []
    print("  test_empty_batch PASS")


def test_stats_after_complete():
    s = ContinuousBatchScheduler()
    r1 = s.submit("a", [1])
    r2 = s.submit("b", [2])
    s.get_batch()
    s.complete(r1, [10], "A")
    s.complete(r2, [20], "B")
    s.get_results()
    stats = s.get_stats()
    assert stats["running"] == 0
    assert stats["pending"] == 0
    print("  test_stats_after_complete PASS")


def test_concurrent_submit_uses_unique_ids():
    s = ContinuousBatchScheduler()
    ids = []
    ids_lock = threading.Lock()

    def submit_one():
        rid = s.submit("hello", [1, 2], max_new_tokens=2)
        with ids_lock:
            ids.append(rid)

    threads = [threading.Thread(target=submit_one) for _ in range(64)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ids) == 64
    assert len(set(ids)) == 64


def test_pending_queue_is_bounded():
    s = ContinuousBatchScheduler(SchedulerConfig(max_queue_size=2))
    assert s.submit("a", [1]) != s.submit("b", [2])
    try:
        s.submit("c", [3])
    except RuntimeError as exc:
        assert "queue is full" in str(exc)
    else:
        raise AssertionError("third submit should be rejected")
    assert s.get_stats()["pending"] == 2


def test_oversized_request_is_rejected_immediately():
    scheduler = ContinuousBatchScheduler(SchedulerConfig(max_total_tokens=5))
    with pytest.raises(ValueError, match="can never fit"):
        scheduler.submit("too long", [1, 2, 3], max_new_tokens=3)
    assert scheduler.get_stats()["pending"] == 0


def test_running_requests_reserve_their_remaining_token_allowance():
    scheduler = ContinuousBatchScheduler(
        SchedulerConfig(max_batch_size=4, max_total_tokens=6)
    )
    first = scheduler.submit("first", [1, 2], max_new_tokens=2)
    assert [request.id for request in scheduler.get_batch()] == [first]
    scheduler._running[first].generated_tokens.append(3)
    second = scheduler.submit("second", [4], max_new_tokens=2)
    batch = scheduler.get_batch()
    assert [request.id for request in batch] == [first]
    assert scheduler.get_stats()["pending"] == 1
    assert scheduler._pending[0].id == second


def test_get_batch_respects_max_total_tokens():
    cfg = SchedulerConfig(max_batch_size=4, max_total_tokens=6)
    s = ContinuousBatchScheduler(cfg)
    a = s.submit("a", [1, 2], max_new_tokens=2)
    b = s.submit("b", [3, 4], max_new_tokens=2)
    c = s.submit("c", [5], max_new_tokens=1)

    batch = s.get_batch()
    assert {req.id for req in batch} == {a, c}
    assert s.get_stats()["pending"] == 1

    for req in batch:
        s.complete(req.id, [9], "x")
    assert {req.id for req in s.get_batch()} == {b}


if __name__ == "__main__":
    test_scheduler_create()
    test_submit_and_stats()
    test_get_batch_fcfs()
    test_complete_and_results()
    test_cancel()
    test_max_batch_size()
    test_empty_batch()
    test_stats_after_complete()
    test_concurrent_submit_uses_unique_ids()
    test_pending_queue_is_bounded()
    test_oversized_request_is_rejected_immediately()
    test_running_requests_reserve_their_remaining_token_allowance()
    test_get_batch_respects_max_total_tokens()
    print("ALL continuous_batching TESTS PASS")
