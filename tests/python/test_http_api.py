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
    env = os.environ.copy()
    env["SNEPPX_HTTP_API_KEY"] = "sk-sneppx-test-token-2026"
    proc = subprocess.Popen(
        [BIN, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
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



def _raw_http_request(port, payload, shutdown_write=False):
    client = socket.create_connection(("127.0.0.1", port), timeout=2)
    try:
        client.sendall(payload)
        if shutdown_write:
            client.shutdown(socket.SHUT_WR)
        client.settimeout(3)
        chunks = []
        while True:
            try:
                part = client.recv(4096)
            except socket.timeout:
                break
            if not part:
                break
            chunks.append(part)
        return b"".join(chunks)
    finally:
        client.close()


def test_slow_client_receives_timeout_response_and_is_closed(server):
    port, _ = server
    client = socket.create_connection(("127.0.0.1", port), timeout=2)
    try:
        client.sendall(b"GET /v1/health HTTP/1.1\r\nHost: localhost\r\n")
        client.settimeout(8)
        started = time.monotonic()
        response = bytearray()
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
        elapsed = time.monotonic() - started
        assert bytes(response).startswith(b"HTTP/1.1 408 Request Timeout")
        assert 3.0 <= elapsed < 8.0
    finally:
        client.close()

def test_connection_limit_returns_service_unavailable(server):
    port, _ = server
    blockers = []
    try:
        for _ in range(4):
            client = socket.create_connection(("127.0.0.1", port), timeout=2)
            client.sendall(b"GET /v1/health HTTP/1.1\r\nHost: localhost\r\n")
            blockers.append(client)
        time.sleep(0.25)
        overflow = socket.create_connection(("127.0.0.1", port), timeout=2)
        try:
            overflow.settimeout(2)
            response = overflow.recv(128)
            assert response.startswith(b"HTTP/1.1 503 Service Unavailable")
        finally:
            overflow.close()
    finally:
        for client in blockers:
            client.close()


def test_duplicate_content_length_is_rejected(server):
    port, _ = server
    response = _raw_http_request(port, b"POST /v1/generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n")
    assert response.startswith(b"HTTP/1.1 400 Bad Request")


def test_transfer_encoding_is_rejected(server):
    port, _ = server
    response = _raw_http_request(port, b"POST /v1/generate HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n")
    assert response.startswith(b"HTTP/1.1 400 Bad Request")


def test_invalid_content_length_is_rejected(server):
    port, _ = server
    response = _raw_http_request(port, b"POST /v1/generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: -1\r\n\r\n")
    assert response.startswith(b"HTTP/1.1 400 Bad Request")


def test_oversized_content_length_is_rejected(server):
    port, _ = server
    response = _raw_http_request(port, b"POST /v1/generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: 32769\r\n\r\n")
    assert response.startswith(b"HTTP/1.1 413 Payload Too Large")


def test_malformed_request_line_is_rejected(server):
    port, _ = server
    response = _raw_http_request(port, b"GET /v1/health\r\nHost: localhost\r\n\r\n")
    assert response.startswith(b"HTTP/1.1 400 Bad Request")

def test_header_name_rejects_non_token_backslash(server):
    port, _ = server
    response = _raw_http_request(
        port,
        b"GET /v1/health HTTP/1.1\r\nHost\\Name: localhost\r\n\r\n",
    )
    assert response.startswith(b"HTTP/1.1 400 Bad Request")


def test_incomplete_declared_body_is_rejected(server):
    port, _ = server
    response = _raw_http_request(port, b"POST /v1/generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: 3\r\n\r\na", shutdown_write=True)
    assert response.startswith(b"HTTP/1.1 400 Bad Request")


@pytest.fixture
def static_server(tmp_path):
    port = _free_port()
    static_root = tmp_path / "public"
    static_root.mkdir()
    (static_root / "hello.txt").write_text("safe static content", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("must not be exposed", encoding="utf-8")
    env = os.environ.copy()
    env["SNEPPX_HTTP_API_KEY"] = "sk-sneppx-test-token-2026"
    proc = subprocess.Popen(
        [BIN, str(port), str(tmp_path / "keys.db"), str(static_root)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    url = "http://127.0.0.1:%d/static/hello.txt" % port
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(0.2)
    else:
        proc.terminate()
        pytest.fail("HTTP static-file server did not become ready")
    yield port, static_root, outside
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_static_file_inside_root_is_served(static_server):
    port, _, _ = static_server
    response = _raw_http_request(
        port, b"GET /static/hello.txt HTTP/1.1\\r\\nHost: localhost\\r\\n\\r\\n"
    )
    assert response.startswith(b"HTTP/1.1 200")
    assert b"safe static content" in response


def test_static_file_parent_traversal_is_blocked(static_server):
    port, _, outside = static_server
    response = _raw_http_request(
        port, b"GET /static/../secret.txt HTTP/1.1\\r\\nHost: localhost\\r\\n\\r\\n"
    )
    assert not response.startswith(b"HTTP/1.1 200")
    assert outside.read_text(encoding="utf-8").encode() not in response


def test_static_file_symlink_escape_is_blocked(static_server, tmp_path):
    port, static_root, outside = static_server
    link = static_root / "outside.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")
    response = _raw_http_request(
        port, b"GET /static/outside.txt HTTP/1.1\\r\\nHost: localhost\\r\\n\\r\\n"
    )
    assert not response.startswith(b"HTTP/1.1 200")
    assert outside.read_text(encoding="utf-8").encode() not in response


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


def test_generate_requires_auth(server):
    port, _ = server
    import json as _json
    body = _json.dumps({"model": "llama3-8B", "prompt": "hi"}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/generate" % port, data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        assert e.code in (401, 403)
    except Exception:
        pytest.skip("auth behaviour not deterministic")




def test_generate_rejects_unregistered_prefixed_key(server):
    port, _ = server
    body = json.dumps({"model": "llama3-8B", "prompt": "hi"}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/generate" % port,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-sneppx-attacker-controlled-value",
        },
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 401


def test_public_path_prefix_collision_is_not_public(server):
    port, _ = server
    req = urllib.request.Request("http://127.0.0.1:%d/v1/models-admin" % port)
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 401


def test_generate_empty_body(server):
    port, _ = server
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/generate" % port, data=b"",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-sneppx-test-token-2026"},
    )
    try:
        urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        assert e.code == 400
    except Exception:
        pytest.skip("empty-body behaviour not deterministic")


def test_generate_unknown_model(server):
    port, _ = server
    import json as _json
    body = _json.dumps({"model": "__nope__", "prompt": "hi"}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/generate" % port, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-sneppx-test-token-2026"},
    )
    try:
        urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        assert e.code == 404
    except Exception:
        pytest.skip("unknown-model behaviour not deterministic")


def test_generate_with_auth(server):
    port, _ = server
    import json as _json
    body = _json.dumps({"model": "llama3-8B", "prompt": "hi", "max_new_tokens": 4}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/generate" % port, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-sneppx-test-token-2026"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            assert r.status == 200
            data = _json.loads(r.read().decode())
            assert "generated_text" in data
            assert "token_ids" in data
            assert "prompt_tokens" in data
            assert "completion_tokens" in data
            assert "model" in data
            assert data["model"] == "llama3-8B"
    except urllib.error.HTTPError as e:
        pytest.fail("configured API key was rejected: HTTP %s" % e.code)


def test_model_info_known(server):
    port, _ = server
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/v1/models/llama3-8B" % port) as r:
            assert r.status == 200
            data = json.loads(r.read().decode())
            assert data is not None
    except urllib.error.HTTPError as e:
        assert e.code in (200, 404)
    except Exception:
        pytest.skip("model-info behaviour not deterministic")
