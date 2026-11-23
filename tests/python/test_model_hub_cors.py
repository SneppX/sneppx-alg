"""Regression: model hub CORS must use an explicit allowlist (M-17)."""

import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from model_hub.api.main import HubDatabase, create_app, _default_cors_origins


def test_wildcard_origins_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="must not contain"):
        create_app(str(tmp_path / "hub"), cors_origins=["*"])


def test_default_origins_are_loopback(monkeypatch):
    monkeypatch.delenv("SNEPPX_HUB_CORS_ORIGINS", raising=False)
    origins = _default_cors_origins()
    assert "*" not in origins
    assert origins


def test_env_wildcard_rejected(monkeypatch):
    monkeypatch.setenv("SNEPPX_HUB_CORS_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="must not contain"):
        _default_cors_origins()


def test_preflight_only_allowed_for_allowlisted_origin(tmp_path, monkeypatch):
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("SNEPPX_HUB_ADMIN_PASSWORD", "a-strong-unique-passphrase-2026")
    data_dir = tmp_path / "hub"
    HubDatabase(str(data_dir))
    app = create_app(str(data_dir), cors_origins=["https://trusted.example"])
    with TestClient(app) as client:
        headers = {
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        }
        resp = client.options("/api/v1/models", headers=headers)
        assert resp.headers.get("access-control-allow-origin") is None

        headers["Origin"] = "https://trusted.example"
        resp = client.options("/api/v1/models", headers=headers)
        assert resp.headers.get("access-control-allow-origin") == (
            "https://trusted.example"
        )
