"""Hardening tests for audit logging backends."""

import time
import urllib.request

import pytest

from SneppX_ALG.interface_bindings.audit_logger import (
    AuditAction,
    AuditEvent,
    AuditResult,
    AuditSeverity,
    RemoteAPIAuditBackend,
    _NoRedirectHandler,
)


def _event():
    return AuditEvent(
        timestamp=time.time(),
        event_id="evt-1",
        action=AuditAction.LOGIN,
        severity=AuditSeverity.INFO,
        result=AuditResult.SUCCESS,
        actor="alice",
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://audit.example.com/events",
        "ftp://audit.example.com/events",
        "https://user:pass@audit.example.com/events",
        "https://audit.example.com/events?token=secret",
        "https://audit.example.com/events#frag",
    ],
)
def test_remote_endpoint_rejects_unsafe_urls(endpoint):
    with pytest.raises(ValueError):
        RemoteAPIAuditBackend(endpoint=endpoint, api_key="key", flush_interval=1000)


def test_remote_endpoint_allows_https():
    backend = RemoteAPIAuditBackend(
        endpoint="https://audit.example.com/events",
        api_key="key",
        flush_interval=1000,
    )
    try:
        assert backend.endpoint == "https://audit.example.com/events"
    finally:
        backend.close()


def test_remote_buffer_is_bounded():
    backend = RemoteAPIAuditBackend(
        endpoint="https://audit.example.com/events",
        api_key="key",
        batch_size=1000,
        max_buffer=2,
        flush_interval=1000,
    )
    try:
        assert backend.write(_event()) is True
        assert backend.write(_event()) is True
        assert backend.write(_event()) is False
        assert len(backend._buffer) == 2
    finally:
        backend.close()


def test_remote_flush_refuses_redirects(monkeypatch):
    backend = RemoteAPIAuditBackend(
        endpoint="https://audit.example.com/events",
        api_key="secret",
        flush_interval=1000,
    )
    calls = []

    class FakeOpener:
        def open(self, req, timeout):
            calls.append((req, timeout))

    def fake_build_opener(*handlers):
        assert any(isinstance(h, _NoRedirectHandler) for h in handlers)
        return FakeOpener()

    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)
    try:
        backend.write(_event())
        backend._flush_batch()
    finally:
        backend.close()

    assert len(calls) == 1
    assert calls[0][0].get_header("Authorization") == "Bearer secret"
