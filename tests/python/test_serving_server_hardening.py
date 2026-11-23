"""M-15 regressions for the stdlib ServingServer request boundary."""

import http.client
import json

import pytest

from SneppX_ALG.interface_bindings import ServingConfig, ServingEngine, ServingServer


class _FakeHandler:
    """Minimal handler surface exercised by the dispatch closure."""

    def __init__(self, address="127.0.0.1", headers=None):
        self.client_address = (address, 12345)
        self.headers = headers or {}


def _engine(**config):
    cfg = ServingConfig(warmup_iters=0, **config)
    eng = ServingEngine(config=cfg, model_fn=lambda x: {"echo": x})
    eng.register_model("demo", "v1")
    return eng


def _dispatch(server):
    return server._build_handler()._dispatch


def _start(server):
    server.start_background()
    return server


def _request(server, method, path, payload=None, headers=None, raw=None):
    conn = http.client.HTTPConnection(server.host, server.port, timeout=5)
    try:
        body = (
            raw if raw is not None else ("" if payload is None else json.dumps(payload))
        )
        hdrs = dict(headers or {})
        if raw is not None or payload is not None:
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=body, headers=hdrs)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()


@pytest.fixture
def server_factory():
    servers = []

    def make(**kwargs):
        srv = _start(ServingServer(_engine(), **kwargs))
        servers.append(srv)
        return srv

    yield make
    for srv in servers:
        srv.stop()


def test_non_loopback_mutating_request_denied():
    srv = ServingServer(_engine(), auth_token=None)
    dispatch = _dispatch(srv)
    remote = _FakeHandler(address="203.0.113.5")
    status, _, body = dispatch(remote, "POST", "/v1/generate", "{}")
    assert status == 401
    assert json.loads(body)["error"] == "unauthorized"
    # Reads remain available without authentication.
    status, _, _ = dispatch(remote, "GET", "/healthz", "")
    assert status == 200


def test_bearer_token_required_and_constant_time_compared():
    srv = ServingServer(_engine(), auth_token="s3cret")
    dispatch = _dispatch(srv)
    loopback = _FakeHandler()
    response = dispatch(loopback, "POST", "/v1/generate", "{}")
    assert response[0] == 401
    wrong = _FakeHandler(headers={"Authorization": "Bearer wrong"})
    assert dispatch(wrong, "POST", "/v1/generate", "{}")[0] == 401
    auth = _FakeHandler(headers={"Authorization": "Bearer s3cret"})
    assert dispatch(auth, "POST", "/v1/generate", '{"inputs": "hi"}')[0] == 200


def test_body_cap_returns_413(server_factory):
    srv = server_factory(max_body_bytes=64)
    headers = {"Authorization": "Bearer s3cret"}
    status, body = _request(srv, "POST", "/v1/generate", raw="x" * 200, headers=headers)
    assert status == 413
    assert body["error"] == "request body too large"


def test_invalid_json_and_non_object_rejected(server_factory):
    srv = server_factory(auth_token="s3cret")
    headers = {"Authorization": "Bearer s3cret"}
    status, body = _request(srv, "POST", "/v1/generate", raw="{nope", headers=headers)
    assert status == 400
    assert body["error"] == "invalid_json"
    status, body = _request(srv, "POST", "/v1/generate", raw="[1,2]", headers=headers)
    assert status == 400
    assert body["error"] == "expected_object"


def test_deploy_payload_validation(server_factory):
    srv = server_factory(auth_token="s3cret")
    headers = {"Authorization": "Bearer s3cret"}
    bad = [
        {"model": "demo", "version": "v2", "weight": -1},
        {"model": "demo", "version": "v2", "weight": True},
        {"model": "demo", "version": "v2", "promote": "yes"},
        {"model": 7, "version": "v2"},
    ]
    for payload in bad:
        status, body = _request(srv, "POST", "/v1/deploy", payload, headers)
        assert status == 400
        assert body["error"] == "invalid_deploy_request"
    status, body = _request(
        srv,
        "POST",
        "/v1/deploy",
        {"model": "demo", "version": "v2", "weight": 10},
        headers,
    )
    assert status == 200
    assert body["status"] == "deployed"


def test_traffic_split_validation(server_factory):
    srv = server_factory(auth_token="s3cret")
    headers = {"Authorization": "Bearer s3cret"}
    for split in ["v1", "v1:abc", "v1:-5", ""]:
        status, body = _request(
            srv, "POST", "/v1/traffic", {"model": "demo", "split": split}, headers
        )
        assert status == 400
    status, body = _request(
        srv, "POST", "/v1/traffic", {"model": "demo", "split": "v1:100"}, headers
    )
    assert status == 200
    assert body["status"] == "ok"


def test_generate_payload_validation(server_factory):
    srv = server_factory(auth_token="s3cret")
    headers = {"Authorization": "Bearer s3cret"}
    bad = [
        {},
        {"inputs": None},
        {"inputs": "hi", "request_id": 7},
        {"inputs": "hi", "version": 3},
    ]
    for payload in bad:
        status, body = _request(srv, "POST", "/v1/generate", payload, headers)
        assert status == 400
    status, body = _request(srv, "POST", "/v1/generate", {"inputs": "hi"}, headers)
    assert status == 200
    assert body["echo"] == "hi"
