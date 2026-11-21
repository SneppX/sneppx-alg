"""Tests for the vizmon dashboard Python bindings."""

import numpy as np
from SneppX_ALG.interface_bindings.vizmon import (
    VizMon,
    _HAS_C,
    _VIZMON_LIB,
)


def test_create_destroy():
    if not _HAS_C:
        print("  test_create_destroy SKIP (no C backend)")
        return
    vm = VizMon()
    assert vm._handle is not None
    vm.close()
    print("  test_create_destroy PASS")


def test_push_scalar_and_snapshot():
    if not _HAS_C:
        print("  test_push_scalar_and_snapshot SKIP (no C backend)")
        return
    vm = VizMon()
    vm.push_scalar("loss", 0.5, 1.0)
    vm.push_scalar("accuracy", 0.8, 2.0)
    snap = vm.snapshot_json()
    assert "scalars" in snap
    assert "series" in snap
    assert "loss" in snap["series"]
    assert snap["series"]["loss"][-1] == [0.5, 1.0]
    assert "accuracy" in snap["series"]
    assert snap["series"]["accuracy"][-1] == [0.8, 2.0]
    vm.close()
    print("  test_push_scalar_and_snapshot PASS")


def test_graph_nodes():
    if not _HAS_C:
        print("  test_graph_nodes SKIP (no C backend)")
        return
    vm = VizMon()
    vm.add_node("x", [], 0, [4])
    vm.add_node("linear", [0], 1, [4, 8])
    snap = vm.snapshot_json()
    assert len(snap["graph"]) == 2
    assert snap["graph"][0]["name"] == "x"
    assert snap["graph"][1]["name"] == "linear"
    assert snap["graph"][1]["ins"] == [0]
    assert snap["graph"][1]["shape"] == [4.0, 8.0]
    vm.close()
    print("  test_graph_nodes PASS")


def test_embedding_pca():
    if not _HAS_C:
        print("  test_embedding_pca SKIP (no C backend)")
        return
    vm = VizMon()
    pts = np.array([
        [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
        [1.0, 1.1, 0.0], [-1.0, -1.1, 0.0],
        [1.1, 1.0, 0.0], [-0.9, -1.0, 0.0],
    ])
    vm.push_embedding("h", pts)
    vm.project_pca()
    snap = vm.snapshot_json()
    assert "embeddings" in snap
    emb = snap["embeddings"].get("h", {})
    assert emb["n_samples"] == 6
    assert emb["projected"] == 1
    assert "points" in emb
    assert len(emb["points"]) == 6
    vm.close()
    print("  test_embedding_pca PASS")


def test_frontend_html():
    if not _HAS_C:
        print("  test_frontend_html SKIP (no C backend)")
        return
    html = VizMon.get_frontend_html()
    assert "vue" in html.lower()
    assert "chart.js" in html.lower()
    assert "/snapshot" in html
    assert "/ws" in html
    print("  test_frontend_html PASS")


def test_export_html():
    if not _HAS_C:
        print("  test_export_html SKIP (no C backend)")
        return
    vm = VizMon()
    vm.push_scalar("loss", 0.3, 0.0)
    html = vm.export_html()
    assert "const SNAPSHOT" in html
    assert html.count("SNAPSHOT") >= 1
    vm.close()
    print("  test_export_html PASS")


def test_histogram_and_timeline():
    if not _HAS_C:
        print("  test_histogram_and_timeline SKIP (no C backend)")
        return
    vm = VizMon()
    vm.push_histogram("w", np.linspace(0, 1, 100), n_bins=20, ts=1.0)
    vm.push_timeline("gemm", 10.0, 2.5, 1024)
    vm.push_sweep("lr1", 0.12, 0.97, 320.0)
    snap = vm.snapshot_json()
    assert "w" in snap["histograms"]
    assert snap["histograms"]["w"]["n_bins"] == 20
    assert len(snap["timeline"]) == 1
    assert snap["timeline"][0]["name"] == "gemm"
    assert len(snap["sweeps"]) == 1
    assert snap["sweeps"][0]["config_id"] == "lr1"
    vm.close()
    print("  test_histogram_and_timeline PASS")


def test_server_start_stop():
    if not _HAS_C:
        print("  test_server_start_stop SKIP (no C backend)")
        return
    import time
    import urllib.request
    vm = VizMon()
    vm.push_scalar("loss", 0.5, 1.0)
    assert vm.start_server(18300) is None  # void return
    time.sleep(0.3)
    try:
        r = urllib.request.urlopen("http://127.0.0.1:18300/snapshot")
        data = r.read()
        assert b"loss" in data
    finally:
        vm.stop_server()
    vm.close()
    print("  test_server_start_stop PASS")


def test_image_sample():
    if not _HAS_C:
        print("  test_image_sample SKIP (no C backend)")
        return
    vm = VizMon()
    png_bytes = bytes([137, 80, 78, 71, 13, 10, 26, 10])
    vm.push_image("sample1", png_bytes, ts=1.0)
    snap = vm.snapshot_json()
    assert len(snap["samples"]) == 1
    assert snap["samples"][0]["tag"] == "sample1"
    vm.close()
    print("  test_image_sample PASS")


if __name__ == "__main__":
    test_create_destroy()
    test_push_scalar_and_snapshot()
    test_graph_nodes()
    test_embedding_pca()
    test_frontend_html()
    test_export_html()
    test_histogram_and_timeline()
    test_server_start_stop()
    test_image_sample()
    print("\nALL vizmon PYTHON binding TESTS PASS")
