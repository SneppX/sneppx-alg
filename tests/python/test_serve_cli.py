"""Tests for Serve CLI."""

from SneppX_ALG.interface_bindings.serve_cli import main as serve_main


def test_serve_cli_import():
    from SneppX_ALG.interface_bindings import serve_cli
    assert serve_cli is not None


def test_serve_cli_has_main():
    assert callable(serve_main)


def test_serve_cli_allows_loopback_without_auth():
    from SneppX_ALG.interface_bindings.serve_cli import _validate_auth_settings

    _validate_auth_settings("127.0.0.1", "none", None, None)
    _validate_auth_settings("::1", "none", None, None)
    _validate_auth_settings("localhost", "none", None, None)
    _validate_auth_settings("::1%lo0", "none", None, None)
    _validate_auth_settings("[::1%lo0]", "none", None, None)



def test_serve_cli_filters_empty_api_key_entries():
    from SneppX_ALG.interface_bindings.serve_cli import _parse_api_keys

    assert _parse_api_keys("key1,, key2, ,") == ["key1", "key2"]
    assert _parse_api_keys(None) == []


def test_serve_cli_rate_limit_zero_disables_without_defaulting():
    from SneppX_ALG.interface_bindings.serve_cli import _resolve_rate_limit

    assert _resolve_rate_limit(None) == (True, 60)
    assert _resolve_rate_limit(0) == (False, 0)
    assert _resolve_rate_limit(25) == (True, 25)


def test_serve_cli_rejects_negative_rate_limit():
    import pytest
    from SneppX_ALG.interface_bindings.serve_cli import _resolve_rate_limit

    with pytest.raises(ValueError, match="zero or a positive"):
        _resolve_rate_limit(-1)


def test_serve_cli_rejects_remote_bind_without_auth():
    import pytest
    from SneppX_ALG.interface_bindings.serve_cli import _validate_auth_settings

    with pytest.raises(ValueError, match="non-loopback"):
        _validate_auth_settings("0.0.0.0", "none", None, None)
    with pytest.raises(ValueError, match="non-loopback"):
        _validate_auth_settings("192.0.2.10", "none", None, None)



def test_serve_cli_rejects_unsupported_checkpoint():
    from unittest.mock import patch
    import pytest
    from SneppX_ALG.interface_bindings import serve_cli

    with patch("sys.argv", ["sneppx-serve", "--checkpoint", "model.sneppx"]):
        with pytest.raises(SystemExit) as exc:
            serve_cli.main()
    assert exc.value.code == 2


def test_serve_cli_requires_credentials_for_enabled_auth():
    import pytest
    from SneppX_ALG.interface_bindings.serve_cli import _validate_auth_settings

    with pytest.raises(ValueError, match="requires"):
        _validate_auth_settings("0.0.0.0", "bearer", None, None)
    with pytest.raises(ValueError, match="requires"):
        _validate_auth_settings("0.0.0.0", "bearer", " , ", None)
    _validate_auth_settings("0.0.0.0", "bearer", "token", None)
    _validate_auth_settings("0.0.0.0", "api_key", None, "keys.sqlite")


if __name__ == "__main__":
    import sys
    locals_ = {k: v for k, v in locals().items() if k.startswith("test_")}
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        try:
            fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
