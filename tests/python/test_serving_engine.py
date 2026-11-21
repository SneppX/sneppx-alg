"""Tests for the SneppX Python serving engine (versioning, A/B, batching, metrics)."""

import os
import tempfile
import time
import threading
from SneppX_ALG.interface_bindings import (
    ServingEngine, ServingConfig, TrafficSplit, MetricsCollector,
    DynamicBatchQueue, ConfigReloader, ModelVersion, WorkerPool,
)


def _make_engine(batches=8, timeout_ms=5.0, workers=1, warmup=0):
    cfg = ServingConfig(max_batch_size=batches, batch_timeout_ms=timeout_ms,
                        n_workers=workers, warmup_iters=warmup)
    eng = ServingEngine(config=cfg, model_fn=lambda x: x)
    return eng


def test_versioning_rolling_update():
    eng = _make_engine()
    assert eng.register_model("gpt", "v1", "release")
    assert eng.register_model("gpt", "v2", "canary", weight=0)
    assert eng.active["gpt"] == "v1" or eng.route("gpt", "r1") == "v1"
    # deploy v3 canary at 0 weight
    assert eng.deploy("gpt", "v3", weight=0)
    # rollback drops the canary (v3)
    assert eng.rollback("gpt")
    # rollback again drops v2, leaving v1
    assert eng.rollback("gpt")
    assert eng.active["gpt"] == "v1"
    # only v1 remains -> cannot rollback
    assert eng.rollback("gpt") is False
    print("  test_versioning_rolling_update PASS")


def test_ab_traffic_split():
    eng = _make_engine()
    eng.register_model("gpt", "v1", "a")
    eng.register_model("gpt", "v2", "b")
    eng.set_traffic("gpt", [("v1", 70), ("v2", 30)])
    n1 = n2 = 0
    for i in range(10000):
        r = eng.route("gpt", f"req-{i}")
        if r == "v1": n1 += 1
        elif r == "v2": n2 += 1
    assert n1 + n2 == 10000
    assert 6500 < n1 < 7500, f"v1 share {n1}"
    assert 2500 < n2 < 3500, f"v2 share {n2}"
    # promote = 100% to v2
    assert eng.promote("gpt", "v2")
    assert eng.active["gpt"] == "v2"
    assert all(eng.route("gpt", f"z{i}") == "v2" for i in range(100))
    print("  test_ab_traffic_split PASS")


def test_dynamic_batching():
    eng = _make_engine(batches=3, timeout_ms=1000.0)
    assert eng.batcher.pending == 0
    eng.batcher.submit("r1"); eng.batcher.submit("r2")
    assert eng.batcher.pending == 2
    drained = eng.batcher.drain()  # not full, but timeout 1s not elapsed yet
    assert drained == [], "not drained before timeout"
    eng.batcher.submit("r3")
    assert eng.batcher.pending == 3
    drained = eng.batcher.drain()
    assert len(drained) == 3, f"drained {len(drained)}"
    assert [d.request_id for d in drained] == ["r1", "r2", "r3"]
    assert eng.batcher.pending == 0
    print("  test_dynamic_batching PASS")


def test_dynamic_batching_timeout():
    eng = _make_engine(batches=32, timeout_ms=10.0)
    eng.batcher.submit("a")
    time.sleep(0.02)
    drained = eng.batcher.drain()
    assert len(drained) == 1, f"timeout drained {len(drained)}"
    print("  test_dynamic_batching_timeout PASS")


def test_metrics_collector():
    m = MetricsCollector()
    for lat in [50, 100, 200, 500, 800, 1200, 3000]:
        m.record(lat * 1000, tokens=8, error=False)
    m.record(100000, 8, error=True)
    d = m.to_dict()
    assert d["requests"] == 8, d["requests"]
    assert d["errors"] == 1
    assert d["tokens"] == 64
    assert d["latency_ms"]["p50"] > 0
    assert d["latency_ms"]["p99"] >= d["latency_ms"]["p50"]
    prom = m.render_prometheus()
    assert "sneppx_serving_requests_total 8" in prom
    assert "sneppx_latency_ms_count 8" in prom
    assert "sneppx_serving_errors_total 1" in prom
    print("  test_metrics_collector PASS")


def test_serving_engine_metrics_and_health():
    eng = _make_engine(warmup=0)
    eng.set_ready(False)
    assert not eng.ready
    eng.serve("hello", request_id="r1")
    assert eng.metrics.requests == 1
    eng.metrics.record(2000, 4, error=True)
    assert eng.metrics.errors == 1
    eng.set_ready(True)
    assert eng.ready
    h = eng.health()
    assert h["ready"] is True
    assert h["models_loaded"] == 0
    # prometheus + json metrics render
    prom = eng.metrics.render_prometheus()
    assert "sneppx_serving_requests_total" in prom
    j = eng.metrics.to_dict()
    assert "uptime_ms" in j and "throughput_rps" in j
    print("  test_serving_engine_metrics_and_health PASS")


def test_warmup():
    eng = _make_engine(warmup=3)
    eng.warmup()
    assert not eng.ready
    r1 = eng.warmup_tick(100, 1)
    r2 = eng.warmup_tick(100, 1)
    r3 = eng.warmup_tick(100, 1)
    assert not r1 and not r2 and r3
    assert eng.ready
    print("  test_warmup PASS")


def test_config_reload():
    eng = _make_engine()
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("serving:\n  max_batch_size: 32\n  batch_timeout_ms: 50\n  n_workers: 4\n  warmup_iters: 10\n")
        path = f.name
    try:
        eng.set_config_path(path)
        assert eng.reload() is True
        assert eng.config.max_batch_size == 32
        assert eng.config.batch_timeout_ms == 50.0
        assert eng.config.n_workers == 4
        assert eng.config.warmup_iters == 10
        assert eng.batcher.max_batch_size == 32
        assert eng.reload() is False  # unchanged
    finally:
        os.unlink(path)
    print("  test_config_reload PASS")


def test_worker_pool_runs_batch():
    eng = _make_engine(batches=2, timeout_ms=1000.0)
    eng.config.n_workers = 1
    seen = []
    lock = threading.Event()

    def run_batch(items):
        seen.extend([it.request_id for it in items])
        lock.set()

    pool = WorkerPool(eng, run_batch, n_workers=1)
    pool.start()
    eng.batcher.submit("a", "k1")
    eng.batcher.submit("b", "k2")
    lock.wait(timeout=2.0)
    pool.stop()
    assert seen == ["a", "b"], seen
    print("  test_worker_pool_runs_batch PASS")


def test_versions_endpoint_data():
    eng = _make_engine()
    eng.register_model("gpt", "v1", "rel")
    eng.register_model("gpt", "v2", "canary", weight=0)
    vs = eng.versions("gpt")
    assert len(vs) == 2
    assert vs[0]["version_id"] == "v1" and vs[0]["weight"] == 100
    assert vs[1]["version_id"] == "v2" and vs[1]["weight"] == 0
    print("  test_versions_endpoint_data PASS")


if __name__ == "__main__":
    test_versioning_rolling_update()
    test_ab_traffic_split()
    test_dynamic_batching()
    test_dynamic_batching_timeout()
    test_metrics_collector()
    test_serving_engine_metrics_and_health()
    test_warmup()
    test_config_reload()
    test_worker_pool_runs_batch()
    test_versions_endpoint_data()
    print("\nALL serving_engine TESTS PASS")
