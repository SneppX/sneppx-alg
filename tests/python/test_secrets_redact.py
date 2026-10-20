"""Tests for sneppx_serve.secrets_redact (sneppx-alg interface_bindings.secrets_redact)."""

from SneppX_ALG.interface_bindings.secrets_redact import redact_secrets, _MASK


def test_redact_openai_key():
    s = "token is sk-abcdefghijklmnopqrstuvwxyz123456"
    out = redact_secrets(s)
    assert _MASK in out
    assert "sk-" not in out or _MASK in out


def test_redact_aws_key():
    s = "AKIAIOSFODNN7EXAMPLE"
    out = redact_secrets(s)
    assert _MASK in out


def test_redact_github_token():
    s = "ghp_" + "a" * 40
    out = redact_secrets(s)
    assert _MASK in out


def test_redact_long_hex():
    s = "secret=" + "a" * 32
    out = redact_secrets(s)
    assert _MASK in out


def test_redact_pem():
    s = "-----BEGIN RSA PRIVATE KEY-----"
    out = redact_secrets(s)
    assert _MASK in out


def test_redact_key_value_api_key():
    s = 'api_key: "supersecret123"'
    out = redact_secrets(s)
    assert _MASK in out
    assert "supersecret123" not in out


def test_redact_password_short():
    # short password with secret key name must still be redacted
    out = redact_secrets({"password": "hunter2"})
    assert out["password"] == _MASK


def test_redact_bearer():
    s = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz"
    out = redact_secrets(s)
    assert _MASK in out


def test_redact_dict_nested():
    data = {
        "user": "alice",
        "password": "hunter2",
        "nested": {"token": "xyz12345678901234567890"},
    }
    out = redact_secrets(data)
    assert out["user"] == "alice"
    assert out["password"] == _MASK
    assert out["nested"]["token"] == _MASK


def test_redact_list():
    data = ["safe", "sk-abcdefghijklmnopqrstuvwxyz123456"]
    out = redact_secrets(data)
    assert out[0] == "safe"
    assert _MASK in out[1]


def test_preserve_non_secret():
    s = "hello world, nothing secret here"
    out = redact_secrets(s)
    assert out == s


def test_redact_int_passthrough():
    assert redact_secrets(42) == 42
    assert redact_secrets(None) is None
    assert redact_secrets(True) is True
