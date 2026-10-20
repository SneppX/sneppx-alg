"""Tests for sneppx_serve.user_store (sneppx-alg interface_bindings.user_store)."""

import os
import tempfile

from SneppX_ALG.interface_bindings.user_store import UserStore


def _store():
    d = tempfile.mkdtemp()
    return UserStore(db_path=os.path.join(d, "users.db"))


def test_create_user():
    s = _store()
    u = s.create_user("a@b.com", "password123")
    assert u is not None
    assert u["email"] == "a@b.com"


def test_get_user():
    s = _store()
    u = s.create_user("a@b.com", "password123")
    uid = u["user_id"] if "user_id" in u else u["id"]
    got = s.get_user(uid)
    assert got is not None
    assert got["email"] == "a@b.com"


def test_get_user_by_email():
    s = _store()
    s.create_user("a@b.com", "password123")
    u = s.get_user_by_email("a@b.com")
    assert u is not None
    assert u["email"] == "a@b.com"


def test_authenticate_user():
    s = _store()
    s.create_user("a@b.com", "password123")
    u = s.authenticate_user("a@b.com", "password123")
    assert u is not None
    # wrong password
    assert s.authenticate_user("a@b.com", "wrong") is None


def test_update_tier():
    s = _store()
    u = s.create_user("a@b.com", "password123")
    uid = u["user_id"] if "user_id" in u else u["id"]
    ok = s.update_tier(uid, "pro")
    assert ok is True or ok is not False
    u2 = s.get_user(uid)
    assert u2["tier"] == "pro"
