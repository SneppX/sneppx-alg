"""Tests for sneppx_serve.api_key_store (sneppx-alg interface_bindings.api_key_store)."""

import os
import tempfile

from SneppX_ALG.interface_bindings.api_key_store import ApiKeyStore


def _store():
    d = tempfile.mkdtemp()
    return ApiKeyStore(db_path=os.path.join(d, "keys.db"))


def test_create_key():
    s = _store()
    rec = s.create_key(name="mykey", tier="free")
    assert rec is not None
    assert "key" in rec
    assert rec["name"] == "mykey"
    assert rec["tier"] == "free"


def test_validate_key():
    s = _store()
    rec = s.create_key(name="k1", tier="free")
    raw = rec["key"]
    info = s.validate_key(raw)
    assert info is not None
    assert info["name"] == "k1"


def test_validate_invalid_key():
    s = _store()
    assert s.validate_key("sk-sneppx-invalid") is None


def test_list_keys():
    s = _store()
    s.create_key(name="k1", tier="free")
    s.create_key(name="k2", tier="free")
    keys = s.list_keys()
    assert len(keys) == 2


def test_revoke_key():
    s = _store()
    rec = s.create_key(name="k1", tier="free")
    info = s.validate_key(rec["key"])
    key_id = info["id"]
    assert s.revoke_key(key_id) is True
    # after revoke, validate should fail
    assert s.validate_key(rec["key"]) is None


def test_update_key():
    s = _store()
    rec = s.create_key(name="k1", tier="free")
    info = s.validate_key(rec["key"])
    key_id = info["id"]
    ok = s.update_key(key_id, name="renamed")
    assert ok is True
    updated = s.get_key(key_id)
    assert updated is not None
    assert updated["name"] == "renamed"


def test_record_and_get_usage():
    s = _store()
    rec = s.create_key(name="k1", tier="free")
    info = s.validate_key(rec["key"])
    key_id = info["id"]
    s.record_usage(key_id, tokens_in=10, tokens_out=5)
    stats = s.get_usage_stats(key_id)
    assert stats is not None
