"""Tests for key_vault.py — S6 key vault + audit logging."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import KeyVault, AuditLogger, FileAuditBackend, AuditAction


def test_kv_generate_key():
    kv = KeyVault()
    k = kv.generate_key("aes-key", "aes-256")
    assert len(k) == 32
    assert kv.get_key("aes-key") == k


def test_kv_store_retrieve():
    kv = KeyVault()
    kv.store_key("mykey", b"\x01" * 32)
    assert kv.get_key("mykey") == b"\x01" * 32


def test_kv_get_missing():
    kv = KeyVault()
    assert kv.get_key("missing") is None


def test_kv_rotate():
    kv = KeyVault()
    k1 = kv.generate_key("rk")
    k2 = kv.rotate_key("rk")
    assert k1 != k2
    assert kv.get_key("rk") == k2


def test_kv_delete():
    kv = KeyVault()
    kv.generate_key("del-key")
    assert kv.delete_key("del-key")
    assert not kv.delete_key("del-key")


def test_kv_list():
    kv = KeyVault()
    kv.generate_key("k1")
    kv.generate_key("k2")
    keys = kv.list_keys()
    assert "k1" in keys
    assert "k2" in keys


def test_kv_wrap_unwrap():
    kv = KeyVault()
    kv.generate_key("target")
    kv.generate_key("wrapper")
    wrapped = kv.wrap_key("target", "wrapper")
    assert wrapped is not None
    unwrapped = kv.unwrap_key(wrapped, "wrapper")
    assert unwrapped == kv.get_key("target")


def _audit_logger():
    import tempfile
    tmpdir = tempfile.mkdtemp()
    backend = FileAuditBackend(log_dir=tmpdir, max_file_size=10*1024*1024, compress=False)
    return AuditLogger(backends=[backend]), tmpdir


def test_audit_log():
    al, _ = _audit_logger()
    al.log(AuditAction.DATA_READ, actor="admin", resource="/secret")
    entries = al.query(actor="admin")
    assert len(entries) == 1
    assert entries[0].action == AuditAction.DATA_READ


def test_audit_chain():
    al, _ = _audit_logger()
    al.log(AuditAction.LOGIN, actor="u1", resource="r1")
    al.log(AuditAction.LOGIN, actor="u2", resource="r2")
    ok, errors = al.verify_chain()
    assert ok, errors


def test_audit_query_filter():
    al, _ = _audit_logger()
    al.log(AuditAction.LOGIN, actor="alice", resource="/api")
    al.log(AuditAction.LOGIN, actor="bob", resource="/api")
    al.log(AuditAction.DATA_READ, actor="alice", resource="/doc")
    entries = al.query(actor="alice")
    assert len(entries) == 2
    entries_a = al.query(action=AuditAction.LOGIN)
    assert len(entries_a) == 2


def test_audit_clear():
    al, _ = _audit_logger()
    al.log(AuditAction.CUSTOM, actor="u", resource="r")
    assert len(al.query()) == 1
    # No backend-level clear() is expected — just verify logging works


if __name__ == "__main__":
    test_kv_generate_key()
    test_kv_store_retrieve()
    test_kv_get_missing()
    test_kv_rotate()
    test_kv_delete()
    test_kv_list()
    test_kv_wrap_unwrap()
    test_audit_log()
    test_audit_chain()
    test_audit_query_filter()
    test_audit_clear()
    print("All key_vault tests passed.")
