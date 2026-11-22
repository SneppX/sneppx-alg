"""Regression: model hub must not bootstrap a default admin account."""

import ast
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model_hub.api.main import HubDatabase


@pytest.fixture
def db(tmp_path):
    return HubDatabase(str(tmp_path / "hub-data"))


def test_bootstrap_requires_explicit_password(db, monkeypatch):
    monkeypatch.delenv("SNEPPX_HUB_ADMIN_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="SNEPPX_HUB_ADMIN_PASSWORD"):
        db.bootstrap_admin()


@pytest.mark.parametrize("password", ["admin", "password", "changeme", "12345678"])
def test_bootstrap_rejects_known_weak_passwords(db, monkeypatch, password):
    monkeypatch.setenv("SNEPPX_HUB_ADMIN_PASSWORD", password)
    with pytest.raises(RuntimeError, match="weak"):
        db.bootstrap_admin()


def test_bootstrap_accepts_strong_password(db, monkeypatch):
    monkeypatch.setenv("SNEPPX_HUB_ADMIN_PASSWORD", "a-strong-unique-passphrase-2026")
    raw = db.bootstrap_admin()
    assert raw and len(raw) > 20


def test_server_default_binds_loopback():
    source = (ROOT / "model_hub" / "cli" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    host_args = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--host"
    ]
    assert host_args
    defaults = [
        kw.value.value
        for kw in host_args[0].keywords
        if kw.arg == "default" and isinstance(kw.value, ast.Constant)
    ]
    assert defaults == ["127.0.0.1"]


def test_startup_logs_fingerprint_not_raw_key():
    source = (ROOT / "model_hub" / "api" / "main.py").read_text(encoding="utf-8")
    assert "fingerprint:" in source
    assert "Created default admin user + API key:" not in source
