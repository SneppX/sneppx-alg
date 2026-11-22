"""Regression: model hub API keys must not be stored in plaintext or outlive revocation."""

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model_hub.api.main import HubDatabase


def _db_with_user(tmp_path):
    db = HubDatabase(str(tmp_path / "hub-data"))
    user = db.create_user(username="alice")
    return db, user


def test_raw_key_is_returned_once_and_not_persisted(tmp_path):
    db, user = _db_with_user(tmp_path)
    key, raw = db.create_api_key(user.id, scopes=["read"])

    assert raw and "raw_key" not in db._keys[-1]
    assert raw not in (tmp_path / "hub-data" / "keys.json").read_text(encoding="utf-8")
    assert db.verify_api_key(raw).id == user.id
    assert key.prefix == raw[:8]


def test_inactive_key_is_rejected(tmp_path):
    db, user = _db_with_user(tmp_path)
    _, raw = db.create_api_key(user.id)

    db._keys[-1]["is_active"] = False
    db._save_json(db._keys_file, db._keys)
    assert db.verify_api_key(raw) is None


def test_expired_key_is_rejected(tmp_path):
    db, user = _db_with_user(tmp_path)
    _, raw = db.create_api_key(user.id)

    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    db._keys[-1]["expires_at"] = past
    db._save_json(db._keys_file, db._keys)
    assert db.verify_api_key(raw) is None


def test_future_expiry_is_accepted(tmp_path):
    db, user = _db_with_user(tmp_path)
    _, raw = db.create_api_key(user.id)

    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    db._keys[-1]["expires_at"] = future
    db._save_json(db._keys_file, db._keys)
    assert db.verify_api_key(raw).id == user.id


def test_authenticate_api_key_exposes_scopes(tmp_path):
    db, user = _db_with_user(tmp_path)
    _, raw = db.create_api_key(user.id, scopes=["read"])

    result = db.authenticate_api_key(raw)
    assert result is not None
    assert result[1].scopes == ["read"]
