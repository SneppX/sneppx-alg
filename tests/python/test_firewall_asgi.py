"""End-to-end tests for firewall enforcement in the ASGI request pipeline."""

import asyncio
from types import SimpleNamespace

from SneppX_ALG.interface_bindings.firewall import FirewallConfig, FirewallRunner
from SneppX_ALG.interface_bindings.firewall_asgi import FirewallASGIMiddleware


def _scope(headers=None, client_ip="192.0.2.10", method="POST", path="/v1/data"):
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers or [(b"host", b"example.test"), (b"content-type", b"application/json")],
        "client": (client_ip, 12345),
        "server": ("example.test", 80),
    }


def _run(middleware, scope, messages):
    sent = []
    queue = list(messages)

    async def receive():
        if queue:
            return queue.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


def test_asgi_middleware_allows_and_replays_request_body():
    received = []
    security = SimpleNamespace(
        firewall=FirewallRunner(),
        release_concurrent=lambda ip: received.append(("released", ip)),
    )

    async def downstream(scope, receive, send):
        message = await receive()
        received.append(message["body"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = FirewallASGIMiddleware(downstream, lambda: security)
    sent = _run(
        middleware,
        _scope(),
        [{"type": "http.request", "body": b'{"ok":true}', "more_body": False}],
    )
    assert received[0] == b'{"ok":true}'
    assert received[1] == ("released", "192.0.2.10")
    assert sent[0]["status"] == 200


def test_asgi_middleware_blocks_denylisted_source_before_handler():
    called = []
    firewall = FirewallRunner(
        FirewallConfig(network={"denylist": ["203.0.113.0/24"]})
    )
    security = SimpleNamespace(firewall=firewall, release_concurrent=lambda ip: None)

    async def downstream(scope, receive, send):
        called.append(True)

    middleware = FirewallASGIMiddleware(downstream, lambda: security)
    sent = _run(
        middleware,
        _scope(client_ip="203.0.113.8", method="GET"),
        [{"type": "http.request", "body": b"", "more_body": False}],
    )
    assert not called
    assert sent[0]["status"] == 403


def test_asgi_middleware_rejects_oversized_stream_without_buffering_all_chunks():
    called = []
    firewall = FirewallRunner(FirewallConfig(application={"max_body_size": 4}))
    security = SimpleNamespace(firewall=firewall, release_concurrent=lambda ip: None)

    async def downstream(scope, receive, send):
        called.append(True)

    middleware = FirewallASGIMiddleware(downstream, lambda: security)
    sent = _run(
        middleware,
        _scope(),
        [
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"45", "more_body": True},
            {"type": "http.request", "body": b"6789", "more_body": False},
        ],
    )
    assert not called
    assert sent[0]["status"] == 413


def test_asgi_middleware_rejects_conflicting_request_framing():
    called = []
    security = SimpleNamespace(firewall=FirewallRunner(), release_concurrent=lambda ip: None)

    async def downstream(scope, receive, send):
        called.append(True)

    middleware = FirewallASGIMiddleware(downstream, lambda: security)
    scope = _scope(headers=[
        (b"host", b"example.test"),
        (b"content-length", b"4"),
        (b"transfer-encoding", b"chunked"),
    ])
    sent = _run(middleware, scope, [{"type": "http.request", "body": b"", "more_body": False}])
    assert not called
    assert sent[0]["status"] == 400


def test_asgi_middleware_rejects_duplicate_content_length():
    called = []
    security = SimpleNamespace(firewall=FirewallRunner(), release_concurrent=lambda ip: None)

    async def downstream(scope, receive, send):
        called.append(True)

    middleware = FirewallASGIMiddleware(downstream, lambda: security)
    scope = _scope(headers=[
        (b"host", b"example.test"),
        (b"content-length", b"1"),
        (b"content-length", b"2"),
    ])
    sent = _run(middleware, scope, [{"type": "http.request", "body": b"x", "more_body": False}])
    assert not called
    assert sent[0]["status"] == 400


def test_asgi_middleware_releases_concurrency_when_handler_raises():
    released = []
    security = SimpleNamespace(
        firewall=FirewallRunner(),
        release_concurrent=lambda ip: released.append(ip),
    )

    async def downstream(scope, receive, send):
        raise RuntimeError("handler failure")

    middleware = FirewallASGIMiddleware(downstream, lambda: security)

    async def invoke():
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}
        async def send(message):
            pass
        await middleware(_scope(method="GET"), receive, send)

    try:
        asyncio.run(invoke())
    except RuntimeError:
        pass
    else:
        raise AssertionError("downstream failure should propagate")
    assert released == ["192.0.2.10"]


def test_asgi_middleware_bypasses_non_http_scopes():
    called = []
    security = SimpleNamespace(firewall=FirewallRunner(), release_concurrent=lambda ip: None)

    async def downstream(scope, receive, send):
        called.append(scope["type"])

    middleware = FirewallASGIMiddleware(downstream, lambda: security)
    sent = _run(
        middleware,
        {"type": "lifespan"},
        [],
    )
    assert called == ["lifespan"]
    assert sent == []
