"""Regression tests for the gitleaks configuration (audit finding M-19)."""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _parse_paths(text):
    """Extract the quoted TOML strings from the allowlist ``paths`` array."""
    section = text.split("[allowlist]", 1)[1]
    array = section.split("paths = [", 1)[1].split("]", 1)[0]
    return re.findall(r"'''([^']+)'''", array)


def test_gitleaks_config_exists_and_extends_defaults():
    path = _ROOT / ".gitleaks.toml"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "useDefault = true" in text
    assert "[allowlist]" in text


def test_gitleaks_allowlist_is_fixture_only():
    paths = _parse_paths((_ROOT / ".gitleaks.toml").read_text(encoding="utf-8"))
    patterns = [re.compile(p) for p in paths]
    fixtures = [
        "docs/API_REFERENCE.md",
        "sonar-project.properties",
        "tests/python/test_scan_secrets.py",
        "tests/python/__pycache__/x.pyc",
    ]
    for fixture in fixtures:
        assert any(p.search(fixture) for p in patterns), fixture
    # Real source and configuration must never be allowlisted.
    for source in ["scripts/scan_secrets.py", "src/app.py", "pyproject.toml"]:
        assert not any(p.search(source) for p in patterns), source
