"""Tests for sneppx_serve.api_gateway (sneppx-alg interface_bindings.api_gateway)."""

from SneppX_ALG.interface_bindings.api_gateway import (
    RegisterRequest,
    RegisterResponse,
    LoginRequest,
    LoginResponse,
    KeyInfo,
    KeyListResponse,
    KeyCreateRequest,
    UsageResponse,
)


def test_register_request():
    r = RegisterRequest(email="a@b.com", password="password123")
    assert r.email == "a@b.com"
    assert r.password == "password123"


def test_register_request_validation():
    try:
        RegisterRequest(email="a@b.com", password="short")
    except Exception:
        return
    raise AssertionError("short password should fail validation")


def test_register_response():
    r = RegisterResponse(user_id="u1", email="a@b.com", tier="free", api_key="sk-xxx")
    assert r.user_id == "u1"
    assert r.message  # default message present


def test_login_request():
    r = LoginRequest(email="a@b.com", password="password123")
    assert r.email == "a@b.com"


def test_login_response():
    r = LoginResponse(user_id="u1", email="a@b.com", tier="free", token="tok")
    assert r.token == "tok"


def test_key_info():
    k = KeyInfo(
        id="k1",
        key_prefix="sk-abc",
        name="mykey",
        tier="free",
        is_active=True,
        created_at=1.0,
    )
    assert k.is_active is True
    assert k.expires_at is None


def test_key_list_response():
    r = KeyListResponse(keys=[])
    assert r.keys == []


def test_key_create_request():
    r = KeyCreateRequest(name="mykey")
    assert r.name == "mykey"


def test_usage_response():
    r = UsageResponse(requests=5, tokens_in=100, tokens_out=50, avg_response_ms=12.5)
    assert r.requests == 5
    assert r.tokens_in == 100
    assert r.tokens_out == 50
