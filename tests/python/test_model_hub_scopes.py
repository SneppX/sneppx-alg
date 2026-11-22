"""Regression: model hub API key scopes must be enforced on write routes."""

import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from model_hub.api.main import HubDatabase, create_app


def test_read_only_key_cannot_mutate_models(tmp_path):
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    data_dir = tmp_path / "hub"
    db = HubDatabase(str(data_dir))
    user = db.create_user(username="reader")
    _, raw = db.create_api_key(user.id, scopes=["read"])

    app = create_app(str(data_dir))
    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {raw}"}
        assert client.get("/api/v1/keys/me", headers=headers).status_code == 200
        response = client.delete("/api/v1/models/does-not-exist", headers=headers)
        assert response.status_code == 403
        assert "write" in response.json()["detail"]


def test_write_key_can_reach_mutating_route(tmp_path):
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    data_dir = tmp_path / "hub"
    db = HubDatabase(str(data_dir))
    user = db.create_user(username="writer")
    _, raw = db.create_api_key(user.id, scopes=["read", "write"])

    app = create_app(str(data_dir))
    with TestClient(app) as client:
        response = client.delete(
            "/api/v1/models/does-not-exist", headers={"Authorization": f"Bearer {raw}"}
        )
        # Scope check passes; the model itself simply does not exist.
        assert response.status_code == 404


def test_read_only_key_cannot_create_admin_key(tmp_path):
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    data_dir = tmp_path / "hub"
    db = HubDatabase(str(data_dir))
    user = db.create_user(username="reader")
    _, raw = db.create_api_key(user.id, scopes=["read"])

    app = create_app(str(data_dir))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/keys",
            data={"username": "new-admin"},
            headers={"Authorization": f"Bearer {raw}"},
        )
        assert response.status_code == 403
