"""Hardening tests for audit logging backends."""

import time
import socket
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

from SneppX_ALG.interface_bindings.audit_logger import (
    AuditLogger,
    AuditWriteError,
    DatabaseAuditBackend,
    FileAuditBackend,
    AuditAction,
    AuditEvent,
    AuditResult,
    AuditSeverity,
    RemoteAPIAuditBackend,
    _NoRedirectHandler,
    _PinnedHTTPSHandler,
)
from SneppX_ALG.interface_bindings import audit_logger as audit_module


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
    "table_name",
    ["audit_log; DROP TABLE users--", "1audit", "audit.log", "audit log"],
)
def test_database_rejects_invalid_table_names(table_name):
    with pytest.raises(ValueError, match="identifier"):
        DatabaseAuditBackend(":memory:", table_name=table_name)


def test_database_accepts_plain_identifier():
    backend = DatabaseAuditBackend(":memory:", table_name="audit_log_2")
    try:
        assert backend.table_name == "audit_log_2"
        assert backend.write(_event()) is True
    finally:
        backend.close()


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


@pytest.mark.parametrize(
    "endpoint",
    ["https://127.0.0.1/events", "https://10.0.0.8/events",
     "https://[::1]/events", "https://localhost/events", "https://audit.internal/events"],
)
def test_remote_endpoint_rejects_private_hosts(endpoint):
    with pytest.raises(ValueError):
        RemoteAPIAuditBackend(endpoint=endpoint, api_key="key", flush_interval=1000)


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
        backend._flush_batch = lambda: False
        backend.close()


def test_remote_write_flushes_at_batch_threshold(monkeypatch):
    calls, observed = [], []

    def public_dns(host, port, type=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("1.1.1.1", port))]

    class FakeOpener:
        def open(self, req, timeout):
            calls.append((req, timeout))
            return None

    def fake_build_opener(*handlers):
        observed.extend(handlers)
        return FakeOpener()

    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)
    backend = RemoteAPIAuditBackend(
        endpoint="https://audit.example.com/events", api_key="key",
        batch_size=1, flush_interval=1000,
    )
    try:
        assert backend.write(_event()) is True
        assert len(calls) == 1
        assert backend._buffer == []
        assert any(isinstance(h, _NoRedirectHandler) for h in observed)
        assert any(isinstance(h, _PinnedHTTPSHandler) for h in observed)
    finally:
        backend.close()


def test_remote_dns_resolution_rejects_private_addresses(monkeypatch):
    def private_dns(host, port, type=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.0.0.8", port))]

    monkeypatch.setattr(socket, "getaddrinfo", private_dns)
    backend = RemoteAPIAuditBackend(
        endpoint="https://audit.example.com/events", api_key="key",
        batch_size=1, flush_interval=1000,
    )
    try:
        assert backend.write(_event()) is True
        assert len(backend._buffer) == 1
    finally:
        backend.close()


def test_remote_write_hash_chain_is_serialized():
    backend = RemoteAPIAuditBackend(
        endpoint="https://audit.example.com/events", api_key="key",
        batch_size=1000, max_buffer=1000, flush_interval=1000,
    )
    try:
        def write_one(_):
            assert backend.write(_event()) is True
        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(write_one, range(96)))
        queued = backend._buffer
        assert len(queued) == 96
        assert queued[0].prev_hash == "0" * 64
        for previous, current in zip(queued, queued[1:]):
            assert current.prev_hash == previous.hash
    finally:
        backend._flush_batch = lambda: False
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

    def public_dns(host, port, type=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("1.1.1.1", port))]

    def fake_build_opener(*handlers):
        assert any(isinstance(h, _NoRedirectHandler) for h in handlers)
        return FakeOpener()

    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)
    try:
        backend.write(_event())
        backend._flush_batch()
    finally:
        backend.close()

    assert len(calls) == 1
    assert calls[0][0].get_header("Authorization") == "Bearer secret"


class _FailingBackend:
    """Backend that always raises on write."""

    def write(self, event):
        raise OSError("disk unavailable")

    def close(self):
        pass


class _FalseBackend:
    """Backend that reports a rejected write via its return value."""

    def write(self, event):
        return False

    def close(self):
        pass


def test_strict_mode_surfaces_backend_exception():
    logger = AuditLogger(backends=[_FailingBackend()], strict=True)
    with pytest.raises(AuditWriteError, match="_FailingBackend"):
        logger.log(action=AuditAction.LOGIN, actor="alice")


def test_strict_mode_surfaces_false_return():
    logger = AuditLogger(backends=[_FalseBackend()], strict=True)
    with pytest.raises(AuditWriteError, match="_FalseBackend"):
        logger.log(action=AuditAction.LOGIN, actor="alice")


def test_default_mode_still_swallows_backend_failure():
    logger = AuditLogger(backends=[_FailingBackend()])
    # Non-strict behaviour is unchanged: failures must not block the caller.
    event = logger.log(action=AuditAction.LOGIN, actor="alice")
    assert event.actor == "alice"


def test_global_audit_logger_factory_is_strict(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_module, "_global_audit_logger", None)
    logger = audit_module.get_audit_logger(log_dir=str(tmp_path))
    try:
        assert logger.strict is True
    finally:
        logger.close()


def test_strict_mode_allows_successful_writes():
    backend = DatabaseAuditBackend(":memory:")
    try:
        logger = AuditLogger(backends=[backend], strict=True)
        event = logger.log(action=AuditAction.LOGIN, actor="alice")
        assert event.actor == "alice"
    finally:
        backend.close()


def _event_with_id(event_id):
    event = _event()
    event.event_id = event_id
    return event


def test_file_backend_restores_chain_head(tmp_path):
    first_backend = FileAuditBackend(log_dir=str(tmp_path), compress=False)
    try:
        first = _event_with_id("evt-1")
        assert first_backend.write(first) is True
        first_hash = first.hash
    finally:
        first_backend.close()

    second_backend = FileAuditBackend(log_dir=str(tmp_path), compress=False)
    try:
        second = _event_with_id("evt-2")
        assert second_backend.write(second) is True
        assert second.prev_hash == first_hash
    finally:
        second_backend.close()


def test_database_backend_restores_chain_head(tmp_path):
    db_path = str(tmp_path / "audit.db")
    first_backend = DatabaseAuditBackend(db_path)
    try:
        first = _event_with_id("evt-1")
        assert first_backend.write(first) is True
        first_hash = first.hash
    finally:
        first_backend.close()

    second_backend = DatabaseAuditBackend(db_path)
    try:
        second = _event_with_id("evt-2")
        assert second_backend.write(second) is True
        assert second.prev_hash == first_hash
    finally:
        second_backend.close()
