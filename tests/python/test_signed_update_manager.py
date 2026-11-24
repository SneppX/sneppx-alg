"""Fail-closed tests for the Python signed update manager."""

import json
from subprocess import CompletedProcess

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from SneppX_ALG.interface_bindings import container_security
from SneppX_ALG.interface_bindings.container_security import ABOOTPartition, SignedUpdateManager


def _manager(tmp_path):
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    public_key_path = tmp_path / "public.key"
    public_key_path.write_bytes(public_bytes)
    manager = SignedUpdateManager(
        update_dir=str(tmp_path / "updates"),
        public_key_path=str(public_key_path),
    )
    return manager, private_key


def _signed_update(tmp_path, private_key, payload=b"verified update bytes"):
    update_path = tmp_path / "update.zip"
    update_path.write_bytes(payload)
    signature_path = tmp_path / "update.zip.sig"
    signature_path.write_bytes(private_key.sign(payload))
    return update_path, signature_path


def _no_dd(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return CompletedProcess(args, 0)

    monkeypatch.setattr(container_security.subprocess, "run", fake_run)
    monkeypatch.setattr(ABOOTPartition, "switch_slot", lambda self: True)
    return calls


def test_stage_rejects_missing_signature(tmp_path):
    manager, private_key = _manager(tmp_path)
    update_path, _ = _signed_update(tmp_path, private_key)

    ok, message = manager.stage_update(str(update_path), "1.0.0")

    assert ok is False
    assert "signature" in message.lower()


def test_stage_rejects_when_no_trusted_key_configured(tmp_path):
    update_path, signature_path = _signed_update(tmp_path, Ed25519PrivateKey.generate())
    manager = SignedUpdateManager(update_dir=str(tmp_path / "updates"))

    ok, message = manager.stage_update(str(update_path), "1.0.0", str(signature_path))

    assert ok is False
    assert "key" in message.lower()


def test_stage_rejects_invalid_signature(tmp_path):
    manager, _ = _manager(tmp_path)
    update_path, _ = _signed_update(tmp_path, Ed25519PrivateKey.generate())

    ok, message = manager.stage_update(str(update_path), "1.0.0", str(tmp_path / "update.zip.sig"))

    assert ok is False
    assert "signature" in message.lower()


def test_stage_records_verified_metadata(tmp_path):
    manager, private_key = _manager(tmp_path)
    update_path, signature_path = _signed_update(tmp_path, private_key)

    ok, update_id = manager.stage_update(str(update_path), "1.0.0", str(signature_path))

    assert ok is True
    staged = manager.update_dir / "staged" / update_id
    meta = json.loads((staged / "meta.json").read_text())
    assert meta["verified"] is True
    assert meta["signature"] == "signature.sig"
    assert (staged / "signature.sig").exists()


def test_install_rejects_unverified_metadata(tmp_path, monkeypatch):
    manager, private_key = _manager(tmp_path)
    update_path, signature_path = _signed_update(tmp_path, private_key)
    ok, update_id = manager.stage_update(str(update_path), "1.0.0", str(signature_path))
    assert ok is True

    meta_path = manager.update_dir / "staged" / update_id / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["verified"] = False
    meta_path.write_text(json.dumps(meta))

    calls = _no_dd(monkeypatch)
    ok, message = manager.install_update(update_id, ABOOTPartition(root=str(tmp_path / "boot")))

    assert ok is False
    assert "verified" in message.lower()
    assert calls == []


def test_install_reverifies_staged_bytes(tmp_path, monkeypatch):
    manager, private_key = _manager(tmp_path)
    update_path, signature_path = _signed_update(tmp_path, private_key)
    ok, update_id = manager.stage_update(str(update_path), "1.0.0", str(signature_path))
    assert ok is True

    staged_update = manager.update_dir / "staged" / update_id / "update.zip"
    staged_update.write_bytes(b"tampered after staging")

    calls = _no_dd(monkeypatch)
    ok, message = manager.install_update(update_id, ABOOTPartition(root=str(tmp_path / "boot")))

    assert ok is False
    assert "signature" in message.lower()
    assert calls == []


def test_verified_install_reaches_partition_write(tmp_path, monkeypatch):
    manager, private_key = _manager(tmp_path)
    update_path, signature_path = _signed_update(tmp_path, private_key)
    ok, update_id = manager.stage_update(str(update_path), "1.0.0", str(signature_path))
    assert ok is True

    calls = _no_dd(monkeypatch)
    ok, message = manager.install_update(update_id, ABOOTPartition(root=str(tmp_path / "boot")))

    assert ok is True, message
    assert calls
    assert calls[0][0] == "dd"
