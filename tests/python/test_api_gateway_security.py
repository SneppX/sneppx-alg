"""Security regression tests for API gateway login and authorization headers."""

import asyncio
from types import SimpleNamespace

from SneppX_ALG.interface_bindings import api_gateway
from SneppX_ALG.interface_bindings.user_store import UserStore


def test_login_issues_a_real_revocable_session_token(tmp_path, monkeypatch):
    store = UserStore(db_path=str(tmp_path / "users.db"))
    store.create_user("login@example.com", "correct-horse-123")
    monkeypatch.setattr(api_gateway, "get_user_store", lambda: store)

    response = asyncio.run(
        api_gateway.login(api_gateway.LoginRequest(
            email="LOGIN@example.com", password="correct-horse-123"
        ))
    )
    assert response.token.startswith("sxs_")
    assert response.token != "sk-session-placeholder"
    assert store.validate_session_token(response.token)["email"] == "login@example.com"


def test_login_rejects_invalid_password(tmp_path, monkeypatch):
    store = UserStore(db_path=str(tmp_path / "users.db"))
    store.create_user("login-fail@example.com", "correct-horse-123")
    monkeypatch.setattr(api_gateway, "get_user_store", lambda: store)

    try:
        asyncio.run(
            api_gateway.login(api_gateway.LoginRequest(
                email="login-fail@example.com", password="wrong-password"
            ))
        )
    except api_gateway.HTTPException as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("invalid credentials must not create a session")


def test_logout_revokes_the_bearer_session(tmp_path, monkeypatch):
    store = UserStore(db_path=str(tmp_path / "users.db"))
    user = store.create_user("logout@example.com", "correct-horse-123")
    token = store.create_session_token(user["id"])
    monkeypatch.setattr(api_gateway, "get_user_store", lambda: store)

    result = asyncio.run(api_gateway.logout("Bearer " + token))
    assert result == {"status": "logged_out"}
    assert store.validate_session_token(token) is None


def test_self_service_authorization_is_read_from_headers_not_query():
    for route in api_gateway.self_service_router.routes:
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        assert all(param.alias != "Authorization" for param in dependant.query_params)
        assert any(param.alias == "Authorization" for param in dependant.header_params)


def test_session_token_can_authenticate_self_service_user(tmp_path, monkeypatch):
    store = UserStore(db_path=str(tmp_path / "users.db"))
    user = store.create_user("self-service@example.com", "correct-horse-123")
    token = store.create_session_token(user["id"])
    monkeypatch.setattr(api_gateway, "get_user_store", lambda: store)
    monkeypatch.setattr(
        api_gateway, "get_key_manager",
        lambda: SimpleNamespace(validate_key=lambda key: None),
    )
    assert api_gateway._get_user_id_from_key("Bearer " + token) == user["id"]
