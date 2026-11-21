"""Regression tests for bounded security state and safe credential caching."""

import pytest

from SneppX_ALG.interface_bindings.api_key_manager import ApiKeyManager
from SneppX_ALG.interface_bindings.api_key_store import ApiKeyStore
from SneppX_ALG.interface_bindings.security_middleware import (
    OutputVerifier,
    OutputVerifierConfig,
    RateLimitConfig,
    SecurityConfig,
    SecurityMiddleware,
)


def test_output_sanitization_matches_case_insensitively():
    verifier = OutputVerifier(
        OutputVerifierConfig(enabled=True, blocked_topics=["confidential phrase"])
    )
    source = "CONFIDENTIAL PHRASE must not leak"
    assert verifier.check(source) == "blocked"
    sanitized = verifier.sanitize(source)
    assert "CONFIDENTIAL PHRASE" not in sanitized
    assert "[redacted]" in sanitized


def test_security_rate_limiter_fails_closed_when_identity_capacity_is_full():
    middleware = SecurityMiddleware(
        SecurityConfig(rate_limit=RateLimitConfig(enabled=True)),
        max_rate_limiters=1,
    )
    assert middleware.check_rate_limit("user-a")[0]
    allowed, reason = middleware.check_rate_limit("user-b")
    assert not allowed
    assert "capacity" in reason.lower()
    assert middleware.check_rate_limit("user-a")[0]


def test_security_rate_limiter_rejects_invalid_capacity():
    with pytest.raises(ValueError):
        SecurityMiddleware(max_rate_limiters=0)


def _manager(tmp_path, **kwargs):
    store = ApiKeyStore(db_path=str(tmp_path / "keys.db"))
    return ApiKeyManager(store=store, **kwargs)


def test_api_key_cache_is_bounded_and_refetches_evicted_keys(tmp_path):
    manager = _manager(tmp_path, max_cache_entries=1)
    first = manager.create_key(name="first")
    second = manager.create_key(name="second")
    assert manager.validate_key(first["key"]) is not None
    assert manager.validate_key(second["key"]) is not None
    assert len(manager._cache) == 1
    assert manager.validate_key(first["key"]) is not None
    assert len(manager._cache) == 1


def test_cached_key_metadata_is_returned_as_a_copy(tmp_path):
    manager = _manager(tmp_path)
    created = manager.create_key(name="copy")
    first = manager.validate_key(created["key"])
    first["tier"] = "enterprise"
    second = manager.validate_key(created["key"])
    assert second["tier"] == "free"


def test_revocation_invalidates_cached_credentials_immediately(tmp_path):
    manager = _manager(tmp_path)
    created = manager.create_key(name="revoke")
    info = manager.validate_key(created["key"])
    assert manager.revoke_key(info["id"])
    assert manager.validate_key(created["key"]) is None


def test_tier_configuration_is_not_mutable_through_returned_value(tmp_path):
    manager = _manager(tmp_path)
    config = manager.get_tier_config("free")
    config["rate_limit_rpm"] = 1
    assert manager.get_tier_config("free")["rate_limit_rpm"] == 60


def test_invalid_cache_limits_are_rejected(tmp_path):
    store = ApiKeyStore(db_path=str(tmp_path / "keys.db"))
    with pytest.raises(ValueError):
        ApiKeyManager(store=store, cache_ttl=0)
    with pytest.raises(ValueError):
        ApiKeyManager(store=store, max_cache_entries=0)
