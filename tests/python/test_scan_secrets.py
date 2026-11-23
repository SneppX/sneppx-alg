"""Regression tests for the pre-commit secret scanner."""

import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SPEC = importlib.util.spec_from_file_location(
    "scan_secrets", os.path.join(_ROOT, "scripts", "scan_secrets.py")
)
scan_secrets = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scan_secrets)


def _rules(text, **kwargs):
    return [name for _, name in scan_secrets.scan_text(text, **kwargs)]


def test_detects_unambiguous_credential_formats():
    text = "\n".join(
        [
            'KEY = "AKIA1234567890ABCDEF"',
            "ghp_" + "a" * 36,
            "-----BEGIN OPENSSH PRIVATE KEY-----",
        ]
    )
    assert set(_rules(text)) >= {"aws-access-key", "github-token", "pem-private-key"}


def test_generic_assignment_detects_long_secret_but_not_placeholder():
    assert "credential-assignment" in _rules('password = "S9vN2longrandomvalue123"')
    assert "credential-assignment" not in _rules('password = "example-password-1234"')


def test_fixture_mode_skips_generic_assignment_only():
    text = 'password = "correct-horse-123"\nKEY = "AKIA1234567890ABCDEF"'
    names = _rules(text, include_assignments=False)
    assert names == ["aws-access-key"]


def test_main_exit_codes(tmp_path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text('token = "AKIA1234567890ABCDEF"\n', encoding="utf-8")
    assert scan_secrets.main([str(bad)]) == 1
    assert "aws-access-key" in capsys.readouterr().out

    good = tmp_path / "good.py"
    good.write_text('password = "example-password-1234"\n', encoding="utf-8")
    assert scan_secrets.main([str(good)]) == 0
