"""Focused tests for HTTP Authorization header parsing."""

from SneppX_ALG.interface_bindings.security_middleware import (
    AuthConfig,
    Authenticator,
    RateLimitConfig,
    SecurityConfig,
    SecurityMiddleware,
)


def test_bearer_scheme_is_case_insensitive():
    authenticator = Authenticator(
        AuthConfig(mode="bearer", api_keys=["test-token"])
    )

    for scheme in ("Bearer", "bearer", "BEARER", "bEaReR"):
        assert authenticator.authenticate(f"{scheme} test-token") is not None


def test_bearer_scheme_rejects_missing_or_malformed_credentials():
    authenticator = Authenticator(
        AuthConfig(mode="bearer", api_keys=["test-token"])
    )

    for header in ("", "Bearer", "Basic test-token", "Bearer ", "Bearer test token"):
        assert authenticator.authenticate(header) is None


def test_bearer_scheme_rejects_unknown_tokens():
    authenticator = Authenticator(
        AuthConfig(mode="bearer", api_keys=["test-token"])
    )

    assert authenticator.authenticate("Bearer another-token") is None


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
