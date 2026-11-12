"""Focused tests for HTTP Authorization header parsing."""

from SneppX_ALG.interface_bindings.security_middleware import AuthConfig, Authenticator


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
