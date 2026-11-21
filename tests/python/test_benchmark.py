"""Tests for Benchmarking Suite."""

from SneppX_ALG.interface_bindings.benchmark import (
    BenchmarkResult, BenchmarkConfig, BenchmarkTimer,
    BenchmarkSuite, run_benchmark_cli,
)
from SneppX_ALG.interface_bindings.tensor import Tensor
import numpy as np


def test_benchmark_config_defaults():
    cfg = BenchmarkConfig()
    assert cfg.warmup_iters == 10


def test_benchmark_config_custom():
    cfg = BenchmarkConfig(warmup_iters=5, measure_iters=50)
    assert cfg.warmup_iters == 5
    assert cfg.measure_iters == 50


def test_benchmark_timer():
    timer = BenchmarkTimer()
    timer.start()
    elapsed = timer.stop()
    assert elapsed >= 0.0


def test_benchmark_timer_stats():
    timer = BenchmarkTimer()
    timer.start(); timer.stop()
    timer.start(); timer.stop()
    mean, median, std, mn, mx = timer.stats()
    assert mean >= 0.0


def test_benchmark_suite_init():
    suite = BenchmarkSuite()
    assert len(suite.results) == 0


def test_benchmark_suite_benchmark_fn():
    suite = BenchmarkSuite(config=BenchmarkConfig(measure_iters=2, warmup_iters=1, verbose=False))
    result = suite.benchmark("test_op", lambda: 1 + 1)
    assert result.name == "test_op"
    assert result.iterations > 0


def test_benchmark_numpy_fn():
    suite = BenchmarkSuite(config=BenchmarkConfig(measure_iters=2, warmup_iters=1, verbose=False))
    a = np.random.randn(16, 16).astype(np.float32)
    b = np.random.randn(16, 16).astype(np.float32)
    result = suite.benchmark("numpy_matmul", lambda: a @ b)
    assert isinstance(result, BenchmarkResult)


def test_benchmark_suite_print_summary():
    suite = BenchmarkSuite(config=BenchmarkConfig(measure_iters=2, warmup_iters=1, verbose=False))
    suite.benchmark("op", lambda: 1 + 1)
    suite.print_summary()


def test_benchmark_suite_results():
    suite = BenchmarkSuite(config=BenchmarkConfig(measure_iters=2, warmup_iters=1, verbose=False))
    suite.benchmark("op", lambda: 1 + 1)
    assert len(suite.results) == 1


def test_run_benchmark_cli_callable():
    assert callable(run_benchmark_cli)


if __name__ == "__main__":
    import sys
    locals_ = {k: v for k, v in locals().items() if k.startswith("test_")}
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        try:
            fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
