"""Tests for expiring and revocable user session credentials."""

import hashlib
import sqlite3

import pytest

from SneppX_ALG.interface_bindings.user_store import UserStore


def _store(tmp_path):
    return UserStore(db_path=str(tmp_path / "users.db"))


def test_session_tokens_are_random_hashed_and_validated(tmp_path):
    store = _store(tmp_path)
    user = store.create_user("A@Example.COM", "correct-horse-123")
    token = store.create_session_token(user["id"])
    assert token.startswith("sxs_")
    info = store.validate_session_token(token)
    assert info["user_id"] == user["id"]
    assert info["email"] == "a@example.com"
    assert info["token_type"] == "session"

    conn = sqlite3.connect(store._db_path)
    stored_hash = conn.execute(
        "SELECT token_hash FROM user_sessions WHERE user_id = ?", (user["id"],)
    ).fetchone()[0]
    conn.close()
    assert stored_hash == hashlib.sha256(token.encode()).hexdigest()
    assert stored_hash != token


def test_session_token_expiration_is_enforced(tmp_path):
    store = _store(tmp_path)
    user = store.create_user("expiry@example.com", "correct-horse-123")
    token = store.create_session_token(user["id"], expires_in_seconds=60)
    assert store.validate_session_token(token) is not None
    assert store.validate_session_token(token, now=store.validate_session_token(token)["expires_at"]) is None


def test_session_token_revocation_is_immediate(tmp_path):
    store = _store(tmp_path)
    user = store.create_user("revoke@example.com", "correct-horse-123")
    token = store.create_session_token(user["id"])
    assert store.revoke_session_token(token)
    assert store.validate_session_token(token) is None
    assert not store.revoke_session_token(token)


def test_revoke_user_sessions_revokes_all_active_sessions(tmp_path):
    store = _store(tmp_path)
    user = store.create_user("all@example.com", "correct-horse-123")
    first = store.create_session_token(user["id"])
    second = store.create_session_token(user["id"])
    assert store.revoke_user_sessions(user["id"]) == 2
    assert store.validate_session_token(first) is None
    assert store.validate_session_token(second) is None


@pytest.mark.parametrize("ttl", [0, 59, 2_592_001, -1])
def test_invalid_session_lifetime_is_rejected(tmp_path, ttl):
    store = _store(tmp_path)
    user = store.create_user("ttl@example.com", "correct-horse-123")
    with pytest.raises(ValueError):
        store.create_session_token(user["id"], expires_in_seconds=ttl)


def test_inactive_users_cannot_create_sessions(tmp_path):
    store = _store(tmp_path)
    user = store.create_user("inactive@example.com", "correct-horse-123")
    conn = store._get_conn()
    conn.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user["id"],))
    conn.commit()
    with pytest.raises(ValueError):
        store.create_session_token(user["id"])
