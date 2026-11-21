"""Tests for c_thread.py — thread pool kernels."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

import time
from SneppX_ALG.interface_bindings import ThreadPool, WorkStealingPool


def test_thread_pool_start_stop():
    pool = ThreadPool(n_threads=2)
    pool.start()
    pool.stop()
    assert len(pool._workers) == 0


def test_thread_pool_submit():
    results = []
    def worker(x):
        results.append(x)
    pool = ThreadPool(n_threads=2)
    pool.start()
    for i in range(5):
        pool.submit(worker, i)
    pool.wait()
    pool.stop()
    assert len(results) == 5


def test_thread_pool_wait():
    results = []
    def slow_worker(x):
        time.sleep(0.05)
        results.append(x)
    pool = ThreadPool(n_threads=4)
    pool.start()
    for i in range(10):
        pool.submit(slow_worker, i)
    pool.wait()
    pool.stop()
    assert len(results) == 10


def test_thread_pool_n_threads():
    pool = ThreadPool(n_threads=8)
    assert pool.n_threads == 8


def test_work_stealing_start_stop():
    pool = WorkStealingPool(n_threads=2)
    pool.start()
    pool.stop()
    assert len(pool._workers) == 0


def test_work_stealing_submit():
    results = []
    def worker(x):
        results.append(x)
    pool = WorkStealingPool(n_threads=2)
    pool.start()
    for i in range(5):
        pool.submit(worker, i)
    time.sleep(0.2)
    pool.stop()
    assert len(results) == 5


def test_work_stealing_concurrent():
    results_lock = __import__('threading').Lock()
    results = []
    def worker(x):
        with results_lock:
            results.append(x)
    pool = WorkStealingPool(n_threads=4)
    pool.start()
    for i in range(20):
        pool.submit(worker, i)
    time.sleep(0.3)
    pool.stop()
    assert len(results) == 20


if __name__ == "__main__":
    test_thread_pool_start_stop()
    test_thread_pool_submit()
    test_thread_pool_wait()
    test_thread_pool_n_threads()
    test_work_stealing_start_stop()
    test_work_stealing_submit()
    test_work_stealing_concurrent()
    print("All c_thread tests passed.")
