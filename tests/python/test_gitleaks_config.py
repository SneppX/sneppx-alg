"""Regression tests for the gitleaks configuration (audit finding M-19)."""

import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / ".gitleaks.toml"


def _config():
    return tomllib.loads(_CONFIG.read_text(encoding="utf-8"))


def test_gitleaks_config_exists_and_extends_defaults():
    assert _CONFIG.is_file()
    config = _config()
    assert config["extend"]["useDefault"] is True
    assert "allowlist" in config


def test_gitleaks_allowlist_is_fixture_only():
    paths = _config()["allowlist"]["paths"]
    patterns = [re.compile(p) for p in paths]
    fixtures = [
        "docs/API_REFERENCE.md",
        "sonar-project.properties",
        "tests/python/test_scan_secrets.py",
        "tests/python/__pycache__/x.pyc",
        "1.5.3/API_REFERENCE/index.html",
        "1.5.2/search/search_index.json",
    ]
    for fixture in fixtures:
        assert any(p.search(fixture) for p in patterns), fixture
    # Real source and configuration must never be allowlisted.
    for source in ["scripts/scan_secrets.py", "src/app.py", "pyproject.toml"]:
        assert not any(p.search(source) for p in patterns), source
