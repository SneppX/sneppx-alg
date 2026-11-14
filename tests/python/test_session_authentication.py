"""Tests for session-token authentication through SecurityMiddleware."""

from SneppX_ALG.interface_bindings.security_middleware import AuthConfig, Authenticator
from SneppX_ALG.interface_bindings.user_store import UserStore
import SneppX_ALG.interface_bindings.user_store as user_store_module


def test_bearer_authenticator_accepts_active_session_token(tmp_path, monkeypatch):
    store = UserStore(db_path=str(tmp_path / "users.db"))
    user = store.create_user("auth@example.com", "correct-horse-123")
    token = store.create_session_token(user["id"])
    monkeypatch.setattr(user_store_module, "_GLOBAL_USER_STORE", store)

    authenticator = Authenticator(AuthConfig(mode="bearer"))
    info = authenticator.authenticate("Bearer " + token)
    assert info["user_id"] == user["id"]
    assert info["token_type"] == "session"


def test_api_key_mode_accepts_bearer_session_and_rejects_revoked_token(tmp_path, monkeypatch):
    store = UserStore(db_path=str(tmp_path / "users.db"))
    user = store.create_user("api-mode@example.com", "correct-horse-123")
    token = store.create_session_token(user["id"])
    monkeypatch.setattr(user_store_module, "_GLOBAL_USER_STORE", store)

    authenticator = Authenticator(AuthConfig(mode="api_key"))
    assert authenticator.authenticate("Bearer " + token)["user_id"] == user["id"]
    store.revoke_session_token(token)
    assert authenticator.authenticate("Bearer " + token) is None
