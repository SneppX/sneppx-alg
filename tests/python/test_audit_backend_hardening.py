"""Hardening tests for audit logging backends."""

import time
import urllib.request

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
