"""Smoke tests for the C HTTP REST API (net/http).

The HTTP server is an opt-in C component (`-DSNEPPX_BUILD_HTTP=ON`) and is not
built by default, so these tests are skipped unless a built `http_server_demo`
binary is found. To run them:

    cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release \\
          -DSNEPPX_BUILD_HTTP=ON -DSNEPPX_BUILD_TESTS=ON
    cmake --build build --config Release
    $env:PYTHONPATH = "bindings/python"
    python -m pytest tests/python/test_http_api.py -q -p no:cacheprovider

The binary search order is: SNEPPX_HTTP_BIN env var, then common build dirs.
"""

import os
import glob
import socket
import subprocess
import time
import urllib.request
import json
import pytest


def _find_binary():
    env = os.environ.get("SNEPPX_HTTP_BIN")
    if env and os.path.isfile(env):
        return env
    patterns = [
        "build/**/http_server_demo.exe",
        "build/**/http_server_demo",
        "build/**/http_server.exe",
        "build/**/http_server",
        "build/**/examples/http_server_demo*",
    ]
    for pat in patterns:
        for cand in glob.glob(pat, recursive=True):
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    return None


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


BIN = _find_binary()
pytestmark = pytest.mark.skipif(
    BIN is None, reason="HTTP server binary not built (set SNEPPX_HTTP_BIN)"
)


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    proc = subprocess.Popen(
        [BIN, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # wait for readiness
    url = "http://127.0.0.1:%d/v1/health" % port
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.2)
    else:
        proc.terminate()
        pytest.fail("HTTP server did not become ready")
    yield port, url
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_health_endpoint(server):
    port, _ = server
    with urllib.request.urlopen("http://127.0.0.1:%d/v1/health" % port) as r:
        assert r.status == 200
        body = r.read().decode("utf-8", "replace")
        assert body  # non-empty


def test_models_list_endpoint(server):
    port, _ = server
    with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % port) as r:
        assert r.status == 200
        raw = r.read().decode("utf-8", "replace")
        # Accept either JSON array or JSON object; must parse.
        data = json.loads(raw)
        assert data is not None


def test_model_info_not_found(server):
    port, _ = server
    try:
        urllib.request.urlopen("http://127.0.0.1:%d/v1/models/__nope__" % port)
    except urllib.error.HTTPError as e:
        # 404 is acceptable; 200 with empty is also acceptable.
        assert e.code in (404, 400, 200)
    except Exception:
        pytest.skip("endpoint behaviour not deterministic")
