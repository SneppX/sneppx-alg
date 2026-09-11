"""Tests for the benchmarking suite."""

import sys
import os
import tempfile
import json
import csv

# Handle case when __file__ is not defined (e.g., exec from string)
try:
    _base = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _base = os.path.join(
        os.environ.get(
            "ARIX_TEST_DIR", "C:/Users/PC/SneppX/sneppx-alg/tests/python"
        )
    )
sys.path.insert(0, os.path.join(os.path.dirname(_base), "../../bindings/python"))

import numpy as np
from SneppX_ALG.interface_bindings import benchmarking

failed = []


def check(name, cond):
    if not cond:
        print(f"  FAIL {name}")
        failed.append(name)
    else:
        print(f"  PASS {name}")


def test_benchmark_config():
    config = benchmarking.BenchmarkConfig()
    check("default config", config.warmup_iters == 10 and config.measure_iters == 100)
    config2 = benchmarking.BenchmarkConfig(
        warmup_iters=5, measure_iters=50, verbose=False
    )
    check("custom config", config2.warmup_iters == 5 and config2.verbose == False)


def test_benchmark_timer():
    timer = benchmarking.BenchmarkTimer()
    check("timer created", timer is not None)

    timer.start()
    timer.stop()
    timer.start()
    timer.stop()
    check("timer records", len(timer.times) == 2)

    mean, median, std, min_ms, max_ms = timer.stats()
    check("stats computed", all(v >= 0 for v in [mean, median, std, min_ms, max_ms]))
    check("percentile works", timer.percentile(99) >= 0)

    timer.reset()
    check("timer reset", len(timer.times) == 0)


def test_memory_tracker():
    if not hasattr(benchmarking, "MemoryTracker") or not benchmarking.HAS_PSUTIL:
        check("memory tracker skipped", True)
        return

    tracker = benchmarking.MemoryTracker()
    tracker.start()
    # Allocate some memory
    arr = np.zeros(1024 * 1024, dtype=np.float32)
    tracker._sample()
    tracker.stop()
    check("memory tracker works", tracker.peak_mb > 0)


def test_benchmark_suite():
    suite = benchmarking.BenchmarkSuite()
    check("suite created", suite is not None)
    check("device info", "python" in suite._device_info)
    check("has numpy", "numpy" in suite._device_info)


def test_benchmark_matmul():
    suite = benchmarking.BenchmarkSuite()
    # Just test with a small shape to avoid long runtime
    results = suite.benchmark_matmul(shapes=[(32, 32, 32)])
    check("matmul benchmark runs", len(results) == 1)
    check("matmul result structure", hasattr(results[0], "mean_time_ms"))
    check("matmul has metadata", "matmul_32x32x32" in str(results[0].name))


def test_benchmark_conv2d():
    suite = benchmarking.BenchmarkSuite()
    configs = [{"N": 1, "C": 3, "H": 32, "W": 32, "OC": 16, "k": 3, "s": 1}]
    results = suite.benchmark_conv2d(configs=configs)
    check("conv2d benchmark runs", len(results) == 1)
    check("conv2d result", results[0].mean_time_ms >= 0)


def test_benchmark_attention():
    suite = benchmarking.BenchmarkSuite()
    configs = [{"batch": 1, "seq": 64, "dim": 128, "heads": 4}]
    results = suite.benchmark_attention(configs=configs)
    check("attention benchmark runs", len(results) == 1)


def test_benchmark_rnn():
    suite = benchmarking.BenchmarkSuite()
    configs = [
        {"batch": 1, "seq": 32, "input": 64, "hidden": 128, "type": "lstm"},
        {"batch": 1, "seq": 32, "input": 64, "hidden": 128, "type": "gru"},
        {"batch": 1, "seq": 32, "input": 64, "hidden": 128, "type": "rnn"},
    ]
    results = suite.benchmark_rnn(configs=configs)
    check("rnn benchmark runs", len(results) == 3)


def test_benchmark_transformer_block():
    suite = benchmarking.BenchmarkSuite()
    configs = [
        {"batch": 1, "seq": 64, "dim": 256, "heads": 4, "ff_mult": 4, "dropout": 0.1},
        {"batch": 2, "seq": 128, "dim": 512, "heads": 8, "ff_mult": 4, "dropout": 0.1},
    ]
    results = suite.benchmark_transformer_block(configs=configs)
    check("transformer block benchmark runs", len(results) == 2)


def test_benchmark_custom():
    suite = benchmarking.BenchmarkSuite()

    def custom_op():
        x = np.random.randn(100, 100).astype(np.float32)
        return x @ x.T

    result = suite.benchmark("custom_matmul", "custom", custom_op)
    check("custom benchmark runs", result.iterations > 0)
    check("custom benchmark mean", result.mean_time_ms > 0)


def test_benchmark_output_json():
    suite = benchmarking.BenchmarkSuite()
    suite.config.output_dir = tempfile.gettempdir()
    suite.config.save_json = True

    suite.benchmark(
        "test_op", "test", lambda: np.sum(np.random.randn(100, 100).astype(np.float32))
    )

    # Check if JSON was saved
    json_files = [
        f
        for f in os.listdir(tempfile.gettempdir())
        if f.startswith("benchmark_") and f.endswith(".json")
    ]
    check("json saved", len(json_files) > 0)


def test_benchmark_output_csv():
    suite = benchmarking.BenchmarkSuite()
    suite.config.output_dir = tempfile.gettempdir()
    suite.config.save_csv = True

    suite.benchmark(
        "test_op_csv",
        "test",
        lambda: np.sum(np.random.randn(100, 100).astype(np.float32)),
    )

    csv_files = [
        f
        for f in os.listdir(tempfile.gettempdir())
        if f.startswith("benchmark_") and f.endswith(".csv")
    ]
    check("csv saved", len(csv_files) > 0)


def test_device_info():
    suite = benchmarking.BenchmarkSuite()
    info = suite._device_info
    check("device info has platform", "platform" in suite._device_info)
    check("device info has python", "python" in suite._device_info)
    check("device info has numpy", "numpy" in suite._device_info)


def test_warmup_and_measure():
    suite = benchmarking.BenchmarkSuite()
    call_count = [0]

    def counted_fn():
        call_count[0] += 1
        return np.sum(np.random.randn(100, 100).astype(np.float32))

    # Test with minimal config
    suite.config.warmup_iters = 2
    suite.config.measure_iters = 5
    suite.config.min_time_s = 0.0
    suite.config.max_iters = 10

    result = suite.benchmark("counted", "test", counted_fn)
    check("warmup ran", call_count[0] >= 2)
    check("measure ran", result.iterations >= 5)


def test_benchmark_suite_run():
    suite = benchmarking.BenchmarkSuite()
    suite.config.verbose = False

    # Run multiple benchmarks
    suite.benchmark(
        "op1", "cat1", lambda: np.sum(np.random.randn(100, 100).astype(np.float32))
    )
    suite.benchmark(
        "op2", "cat1", lambda: np.mean(np.random.randn(100, 100).astype(np.float32))
    )
    suite.benchmark(
        "op3", "cat2", lambda: np.max(np.random.randn(100, 100).astype(np.float32))
    )

    check("multiple benchmarks", len(suite.results) == 3)
    check("categories tracked", len(set(r.category for r in suite.results)) == 2)


def test_benchmark_result_serialization():
    suite = benchmarking.BenchmarkSuite()
    result = suite.benchmark("serialize_test", "test", lambda: 1.0)

    # Check result can be serialized
    import json

    data = {
        "name": result.name,
        "category": result.category,
        "iterations": result.iterations,
        "total_time_s": result.total_time_s,
        "mean_time_ms": result.mean_time_ms,
        "median_time_ms": result.median_time_ms,
        "std_time_ms": result.std_time_ms,
        "min_time_ms": result.min_time_ms,
        "max_time_ms": result.max_time_ms,
        "throughput_ops_s": result.throughput_ops_s,
        "memory_mb": result.memory_mb,
    }
    json_str = json.dumps(data)
    parsed = json.loads(json_str)
    check("serialization works", parsed["name"] == "serialize_test")


if __name__ == "__main__":
    test_benchmark_config()
    test_benchmark_timer()
    test_memory_tracker()
    test_benchmark_suite()
    test_benchmark_matmul()
    test_benchmark_conv2d()
    test_benchmark_attention()
    test_benchmark_rnn()
    test_benchmark_transformer_block()
    test_benchmark_custom()
    test_benchmark_output_json()
    test_benchmark_output_csv()
    test_device_info()
    test_warmup_and_measure()
    test_benchmark_suite_run()
    test_benchmark_result_serialization()

    print(
        f"\n{'All benchmark tests passed!' if not failed else f'{len(failed)} failures: {failed}'}"
    )
