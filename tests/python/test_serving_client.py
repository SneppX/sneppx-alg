"""End-to-end tests for the SneppX serving client + stdlib ServingServer."""

import json
import time
import threading
from SneppX_ALG.interface_bindings import (
    ServingEngine, ServingConfig, ServingServer, ServingClient,
)


def _start_server(model_fn=None):
    cfg = ServingConfig(max_batch_size=4, batch_timeout_ms=5.0, n_workers=1, warmup_iters=0)
    eng = ServingEngine(config=cfg, model_fn=model_fn or (lambda x: {"out": str(x)[-1:]}))
    srv = ServingServer(eng, host="127.0.0.1", port=0)
    srv.start_background()
    # wait for readiness
    client = ServingClient(base_url=srv.url)
    client.wait_ready(timeout=5.0)
    return eng, srv, client


def test_health_and_ready_endpoints():
    eng, srv, client = _start_server()
    try:
        h = client.health()
        assert h["status"] == "ok"
        assert h["models_loaded"] == 0
        assert client.healthz() is True
        assert client.readyz() is True
    finally:
        srv.stop()
    print("  test_health_and_ready_endpoints PASS")


def test_metrics_endpoint():
    eng, srv, client = _start_server()
    try:
        # record some traffic through the engine directly
        eng.serve("hello", request_id="m1")
        eng.serve("world", request_id="m2", version=None)
        # prometheus text
        s, data = client._request("GET", "/metrics", headers={"Accept": "text/plain"})
        assert "sneppx_serving_requests_total" in (data.get("raw") or "")
        # json metrics
        s2, data2 = client._request("GET", "/metrics", headers={"Accept": "application/json"})
        assert data2["requests"] >= 2
        assert "latency_ms" in data2
    finally:
        srv.stop()
    print("  test_metrics_endpoint PASS")


def test_versions_and_deploy():
    eng, srv, client = _start_server()
    try:
        eng.deploy("m", "v1", "release", weight=100, promote=True)
        eng.deploy("m", "v2", "canary", weight=0)
        versions = client.versions("m")
        assert len(versions) == 2
        assert versions[0]["version_id"] == "v1" and versions[0]["active"] is True
        assert versions[1]["version_id"] == "v2"
        # deploy a new canary via HTTP
        resp = client.deploy("m", "v3", weight=0)
        assert resp["status"] == "deployed"
        versions = client.versions("m")
        assert len(versions) == 3
        # models list
        models = client.models()
        assert any(x["name"] == "m" for x in models)
    finally:
        srv.stop()
    print("  test_versions_and_deploy PASS")


def test_traffic_split_via_client():
    eng, srv, client = _start_server()
    try:
        eng.deploy("m", "v1", weight=100, promote=True)
        eng.deploy("m", "v2", weight=0)
        resp = client.set_traffic("m", "v1:60,v2:40")
        assert resp["status"] == "ok"
        n1 = n2 = 0
        for i in range(10000):
            ver, _ = client.generate("m", i, request_id=f"r{i}")
            if ver == "v1":
                n1 += 1
            else:
                n2 += 1
        assert n1 + n2 == 10000
        assert 5500 < n1 < 6500, f"v1 share {n1}"
        assert 3500 < n2 < 4500, f"v2 share {n2}"
    finally:
        srv.stop()
    print("  test_traffic_split_via_client PASS")


def test_generate_and_error_counters():
    eng, srv, client = _start_server(model_fn=lambda x: {"echo": x})
    eng.set_ready(True)
    try:
        ver, out = client.generate("echo-model", {"msg": "hi"}, request_id="g1")
        assert out["echo"] == {"msg": "hi"}
        s, data = client._request("GET", "/metrics", headers={"Accept": "application/json"})
        assert data["requests"] >= 1
    finally:
        srv.stop()
    print("  test_generate_and_error_counters PASS")


def test_config_hot_reload_over_http():
    import tempfile, os
    cfg = ServingConfig(max_batch_size=4, warmup_iters=0)
    eng = ServingEngine(config=cfg, model_fn=lambda x: x)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("serving:\n  max_batch_size: 64\n  n_workers: 3\n")
        path = f.name
    try:
        eng.set_config_path(path)
        srv = ServingServer(eng, host="127.0.0.1", port=0)
        srv.start_background()
        client = ServingClient(base_url=srv.url)
        client.wait_ready(timeout=5.0)
        # rewrite config
        with open(path, "w") as f:
            f.write("serving:\n  max_batch_size: 128\n  n_workers: 5\n")
        # poll reload
        import time as _t
        deadline = _t.time() + 2.0
        while _t.time() < deadline:
            if eng.reload():
                break
            _t.sleep(0.05)
        assert eng.config.max_batch_size == 128, eng.config.max_batch_size
        assert eng.config.n_workers == 5
        srv.stop()
    finally:
        os.unlink(path)
    print("  test_config_hot_reload_over_http PASS")


if __name__ == "__main__":
    test_health_and_ready_endpoints()
    test_metrics_endpoint()
    test_versions_and_deploy()
    test_traffic_split_via_client()
    test_generate_and_error_counters()
    test_config_hot_reload_over_http()
    print("\nALL serving_client TESTS PASS")
