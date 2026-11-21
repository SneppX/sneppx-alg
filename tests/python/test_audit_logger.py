"""Tests for Audit Logger (S6 Security)."""

from SneppX_ALG.interface_bindings.audit_logger import (
    AuditSeverity, AuditAction, AuditResult, AuditEvent,
    AuditBackend, FileAuditBackend,
    AuditLogger, get_audit_logger, set_audit_logger,
    audit_log, audit_login, audit_data_access,
    audit_config_change, audit_security_event,
    query_audit_log, verify_audit_chain, export_audit_log,
)
import tempfile
import os
import time


def test_audit_severity_values():
    assert AuditSeverity.INFO.value == 1
    assert AuditSeverity.CRITICAL.value == 4


def test_audit_action_values():
    assert AuditAction.LOGIN.value == "login"
    assert AuditAction.DATA_READ.value == "data_read"


def test_audit_result_values():
    assert AuditResult.SUCCESS.value == "success"
    assert AuditResult.FAILURE.value == "failure"


def test_audit_event_creation():
    event = AuditEvent(
        timestamp=time.time(),
        event_id="evt_001",
        action=AuditAction.LOGIN,
        severity=AuditSeverity.INFO,
        result=AuditResult.SUCCESS,
        actor="test_user",
    )
    assert event.actor == "test_user"
    assert event.action == AuditAction.LOGIN


def test_audit_event_to_dict():
    event = AuditEvent(
        timestamp=time.time(),
        event_id="evt_002",
        severity=AuditSeverity.WARNING,
        action=AuditAction.DATA_WRITE,
        actor="admin",
        result=AuditResult.SUCCESS,
        details={"key": "value"},
    )
    d = event.to_dict()
    assert d["actor"] == "admin"
    assert d["details"]["key"] == "value"


def test_audit_logger_init():
    logger = AuditLogger()
    assert logger is not None


def test_audit_logger_log():
    logger = AuditLogger()
    event = logger.log(AuditAction.LOGIN, "user1",
                       resource="/api/login", result=AuditResult.SUCCESS)
    assert event is not None


def test_audit_logger_tamper_chain():
    logger = AuditLogger()
    e1 = logger.log(AuditAction.LOGIN, "u1", result=AuditResult.SUCCESS)
    e2 = logger.log(AuditAction.LOGOUT, "u1", result=AuditResult.SUCCESS)
    assert e2.prev_hash == e1.hash


def test_audit_logger_verify():
    logger = AuditLogger()
    logger.log(AuditAction.LOGIN, "u1", result=AuditResult.SUCCESS)
    valid, errors = logger.verify_chain()
    assert valid is True


def test_file_audit_backend():
    import tempfile as tf
    import os
    with tf.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        os.unlink(path)
    except OSError:
        pass
    try:
        backend = FileAuditBackend(path)
        event = AuditEvent(
            timestamp=time.time(),
            event_id="evt_f1",
            action=AuditAction.LOGIN,
            severity=AuditSeverity.INFO,
            result=AuditResult.SUCCESS,
            actor="user1",
        )
        result = backend.write(event)
        assert result is True
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_get_audit_logger():
    logger = get_audit_logger()
    assert logger is not None


def test_set_audit_logger():
    new_logger = AuditLogger()
    set_audit_logger(new_logger)
    assert get_audit_logger() is new_logger


def test_audit_log():
    event = audit_log(AuditAction.LOGIN, "tester", resource="/system")
    assert event is not None


def test_audit_login_success():
    audit_login("tester", success=True)


def test_audit_login_failure():
    audit_login("tester", success=False)


def test_audit_data_access():
    audit_data_access("tester", AuditAction.DATA_READ,
                      resource="/data/dataset", resource_type="dataset",
                      resource_id="v1", success=True)


def test_audit_config_change():
    audit_config_change("admin", "max_connections", "100", "200", success=True)


def test_audit_security_event():
    audit_security_event(AuditAction.LOGIN_FAILED, "siem_bot",
                         severity=AuditSeverity.WARNING, details={"reason": "test"})


def test_query_audit_log():
    results = query_audit_log()
    assert isinstance(results, list)


def test_verify_audit_chain():
    result = verify_audit_chain()
    assert isinstance(result, tuple)


def test_export_audit_log():
    data = export_audit_log(format="json")
    assert isinstance(data, str)


if __name__ == "__main__":
    import sys
    locals_ = {k: v for k, v in locals().items() if k.startswith("test_")}
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        try:
            fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
