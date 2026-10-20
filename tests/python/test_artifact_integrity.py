"""Tests for sneppx_serve.artifact_integrity (sneppx-alg interface_bindings.artifact_integrity)."""

import os
import tempfile

from SneppX_ALG.interface_bindings.crypto_sign import Ed25519
from SneppX_ALG.interface_bindings.artifact_integrity import (
    sign_artifact,
    verify_artifact,
    sign_file,
    verify_file,
    verify_or_quarantine,
    ArtifactIntegrityError,
)


def _keys():
    sk, pk = Ed25519.keypair()
    return pk, sk


def test_sign_verify_roundtrip():
    pk, sk = _keys()
    data = b"hello artifact"
    bundle = sign_artifact(data, sk, key_id="k1")
    assert bundle["alg"] == "ed25519"
    assert bundle["key_id"] == "k1"
    assert "signature" in bundle
    assert verify_artifact(data, pk, bundle) is True


def test_verify_wrong_data():
    pk, sk = _keys()
    data = b"hello"
    bundle = sign_artifact(data, sk)
    assert verify_artifact(b"tampered", pk, bundle) is False


def test_verify_wrong_key():
    pk, sk = _keys()
    pk2, _ = _keys()
    data = b"hello"
    bundle = sign_artifact(data, sk)
    assert verify_artifact(data, pk2, bundle) is False


def test_verify_bad_bundle():
    pk, _ = _keys()
    assert verify_artifact(b"x", pk, {}) is False
    assert verify_artifact(b"x", pk, {"alg": "rsa"}) is False
    assert verify_artifact(b"x", pk, {"alg": "ed25519", "signature": "zzzz"}) is False


def test_sign_file_verify_file(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"payload bytes")
    pk, sk = _keys()
    bundle = sign_file(str(path), sk, str(path) + ".sig", key_id="k")
    assert os.path.exists(str(path) + ".sig")
    assert verify_file(str(path), pk) is True
    assert verify_file(str(path), pk, str(path) + ".sig") is True


def test_verify_file_missing_sig(tmp_path):
    path = tmp_path / "nope.bin"
    path.write_bytes(b"x")
    pk, _ = _keys()
    assert verify_file(str(path), pk) is False


def test_verify_or_quarantine_ok(tmp_path):
    path = tmp_path / "good.bin"
    path.write_bytes(b"good data")
    qdir = tmp_path / "quarantine"
    pk, sk = _keys()
    sign_file(str(path), sk, str(path) + ".sig")
    assert verify_or_quarantine(str(path), pk, str(qdir)) is True
    assert path.exists()


def test_verify_or_quarantine_bad(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"bad data")
    sig = path.with_suffix(".bin.sig")
    # sign different data so verification fails
    pk, sk = _keys()
    sign_artifact(b"other", sk)
    # write a bogus sig
    import json

    with open(str(sig), "w") as f:
        json.dump({"alg": "ed25519", "signature": "00" * 64}, f)
    qdir = tmp_path / "quarantine"
    result = verify_or_quarantine(str(path), pk, str(qdir))
    assert result is False
    assert not path.exists()
    assert (qdir / "bad.bin").exists()
