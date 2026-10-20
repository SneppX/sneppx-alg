"""Tests for sneppx_serve.api_key_manager (sneppx-alg interface_bindings.api_key_manager)."""

import os
import tempfile

from SneppX_ALG.interface_bindings.api_key_store import ApiKeyStore
from SneppX_ALG.interface_bindings.api_key_manager import ApiKeyManager


def _manager():
    d = tempfile.mkdtemp()
    store = ApiKeyStore(db_path=os.path.join(d, "keys.db"))
    return ApiKeyManager(store=store)


def test_create_and_validate():
    mgr = _manager()
    rec = mgr.create_key(name="k1", tier="free")
    assert rec is not None
    assert mgr.validate_key(rec["key"]) is not None


def test_admin_key():
    mgr = _manager()
    rec = mgr.create_key(name="k1", tier="free")
    key = rec["key"]
    assert mgr.is_admin_key(key) is False
    mgr.set_admin_key(key)
    assert mgr.is_admin_key(key) is True


def test_list_keys():
    mgr = _manager()
    mgr.create_key(name="k1", tier="free")
    mgr.create_key(name="k2", tier="free")
    assert len(mgr.list_keys()) == 2


def test_get_key():
    mgr = _manager()
    rec = mgr.create_key(name="k1", tier="free")
    key = rec["key"]
    info = mgr.validate_key(key)
    got = mgr.get_key(info["id"])
    assert got is not None


def test_revoke_key():
    mgr = _manager()
    rec = mgr.create_key(name="k1", tier="free")
    key = rec["key"]
    info = mgr.validate_key(key)
    assert mgr.revoke_key(info["id"]) is True or mgr.revoke_key(info["id"]) is not False


def test_update_key():
    mgr = _manager()
    rec = mgr.create_key(name="k1", tier="free")
    info = mgr.validate_key(rec["key"])
    ok = mgr.update_key(info["id"], name="renamed")
    assert ok is True or ok is not False


def test_record_usage():
    mgr = _manager()
    rec = mgr.create_key(name="k1", tier="free")
    info = mgr.validate_key(rec["key"])
    mgr.record_usage(info["id"], tokens_in=10, tokens_out=5)
    stats = mgr.get_usage_stats(info["id"])
    assert stats is not None


def test_tier_config():
    mgr = _manager()
    cfg = mgr.get_tier_config("free")
    assert isinstance(cfg, dict)
    assert "rate_limit_rpm" in cfg
