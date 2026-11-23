"""Focused tests for HTTP Authorization header parsing."""

import pytest

from SneppX_ALG.interface_bindings.security_middleware import (
    AuthConfig,
    Authenticator,
    RateLimitConfig,
    SecurityConfig,
    SecurityMiddleware,
)
from SneppX_ALG.interface_bindings.serving_client import ServingClient


def test_bearer_scheme_is_case_insensitive():
    authenticator = Authenticator(AuthConfig(mode="bearer", api_keys=["test-token"]))

    for scheme in ("Bearer", "bearer", "BEARER", "bEaReR"):
        assert authenticator.authenticate(f"{scheme} test-token") is not None


def test_bearer_scheme_rejects_missing_or_malformed_credentials():
    authenticator = Authenticator(AuthConfig(mode="bearer", api_keys=["test-token"]))

    for header in ("", "Bearer", "Basic test-token", "Bearer ", "Bearer test token"):
        assert authenticator.authenticate(header) is None


def test_bearer_scheme_rejects_unknown_tokens():
    authenticator = Authenticator(AuthConfig(mode="bearer", api_keys=["test-token"]))

    assert authenticator.authenticate("Bearer another-token") is None


def test_static_key_uses_constant_time_comparison(monkeypatch):
    import SneppX_ALG.interface_bindings.security_middleware as middleware_module

    real_compare = middleware_module.hmac.compare_digest
    calls = []

    def spy_compare(a, b):
        calls.append((a, b))
        return real_compare(a, b)

    monkeypatch.setattr(middleware_module.hmac, "compare_digest", spy_compare)
    authenticator = Authenticator(AuthConfig(mode="api_key", api_keys=["secret-key"]))

    assert authenticator.authenticate("secret-key") is not None
    assert authenticator.authenticate("wrong-key") is None
    assert calls


def test_static_key_uses_configured_rate_limit():
    middleware = SecurityMiddleware(
        SecurityConfig(
            auth=AuthConfig(mode="bearer", api_keys=["limited-token"]),
            rate_limit=RateLimitConfig(enabled=True, requests_per_minute=1),
        )
    )
    info = middleware.authenticate("Bearer limited-token")
    assert info is not None

    first = middleware.check_rate_limit(
        info["user_id"],
        tier=info["tier"],
        rpm_override=info.get("rate_limit_rpm"),
    )
    second = middleware.check_rate_limit(
        info["user_id"],
        tier=info["tier"],
        rpm_override=info.get("rate_limit_rpm"),
    )
    assert first == (True, "")
    assert second[0] is False
    assert "1/minute" in second[1]


def test_key_specific_rate_limit_overrides_tier_default():
    middleware = SecurityMiddleware(
        SecurityConfig(rate_limit=RateLimitConfig(enabled=True, requests_per_minute=20))
    )

    assert middleware.check_rate_limit("managed-user", "pro", 1) == (True, "")
    allowed, reason = middleware.check_rate_limit("managed-user", "pro", 1)
    assert allowed is False
    assert "1/minute" in reason


def test_rate_limiter_is_atomic_under_concurrent_requests():
    from concurrent.futures import ThreadPoolExecutor

    middleware = SecurityMiddleware(
        SecurityConfig(rate_limit=RateLimitConfig(enabled=True, requests_per_minute=5))
    )
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(
            executor.map(
                lambda _: middleware.check_rate_limit("concurrent-user", "static"),
                range(40),
            )
        )

    assert sum(1 for allowed, _ in results if allowed) == 5


def test_disabled_rate_limiter_allows_requests():
    middleware = SecurityMiddleware(
        SecurityConfig(rate_limit=RateLimitConfig(enabled=False, requests_per_minute=1))
    )
    assert middleware.check_rate_limit("user", "static") == (True, "")
    assert middleware.check_rate_limit("user", "static") == (True, "")


@pytest.mark.parametrize("url", ["http://api.example.com", "http://127.0.0.1.evil"])
def test_serving_client_rejects_api_key_over_remote_http(url):
    with pytest.raises(ValueError, match="HTTPS"):
        ServingClient(base_url=url, api_key="test-api-key")


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8300",
        "http://127.0.0.1:8300",
        "http://[::1]:8300",
        "https://api.example.com",
    ],
)
def test_serving_client_allows_api_key_over_tls_or_loopback(url):
    client = ServingClient(base_url=url, api_key="test-api-key")
    assert client.api_key == "test-api-key"


def test_serving_client_allows_remote_http_without_api_key():
    client = ServingClient(base_url="http://api.example.com")
    assert client.base_url == "http://api.example.com"
