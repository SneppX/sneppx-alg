import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.crypto_sign import Ed25519
from SneppX_ALG.interface_bindings.artifact_integrity import (
    sign_artifact, verify_artifact, sign_file, verify_file, verify_or_quarantine,
)
from SneppX_ALG.interface_bindings.secrets_redact import redact_secrets


def test_sign_verify_roundtrip():
    sk, pk = Ed25519.keypair(seed=os.urandom(32))
    data = np.random.randn(4, 4).tobytes()
    bundle = sign_artifact(data, sk, key_id="k1")
    assert verify_artifact(data, pk, bundle) is True
    # tamper
    assert verify_artifact(data + b"x", pk, bundle) is False
    # wrong key
    _, pk2 = Ed25519.keypair(seed=os.urandom(32))
    assert verify_artifact(data, pk2, bundle) is False
    print("artifact sign/verify: OK")


def test_file_quarantine():
    sk, pk = Ed25519.keypair(seed=os.urandom(32))
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "model.bin")
    with open(path, "wb") as f:
        f.write(b"weights-bytes-1234")
    sig = os.path.join(tmp, "model.bin.sig")
    sign_file(path, sk, sig)
    assert verify_file(path, pk) is True
    # tamper with the artifact -> verify fails
    with open(path, "ab") as f:
        f.write(b"!")
    assert verify_file(path, pk) is False
    qdir = os.path.join(tmp, "quarantine")
    ok = verify_or_quarantine(path, pk, qdir)
    assert ok is False
    assert not os.path.exists(path)
    assert len(os.listdir(qdir)) == 2  # moved artifact + sig
    print("artifact quarantine: OK")


def test_redact_secrets():
    rec = {
        "api_key": "sk-1234567890abcdef",
        "password": "hunter2",
        "loss": 0.123,
        "nested": {"token": "ghp_abcdefghijklmnopqrstuvwxyz"},
        "log": 'Authorization: Bearer abcdef0123456789 "x": 1',
    }
    out = redact_secrets(rec)
    s = str(out)
    assert "sk-1234567890abcdef" not in s
    assert "hunter2" not in s
    assert "ghp_abcdefghijklmnopqrstuvwxyz" not in s
    assert "abcdef0123456789" not in s
    assert "***REDACTED***" in s
    assert out["loss"] == 0.123  # non-secret preserved
    # string form
    txt = redact_secrets('apikey="AKIAIOSFODNN7EXAMPLE" x=1')
    assert "AKIAIOSFODNN7EXAMPLE" not in txt
    assert "***REDACTED***" in txt
    print("redact_secrets: OK")


if __name__ == "__main__":
    test_sign_verify_roundtrip()
    test_file_quarantine()
    test_redact_secrets()
    print("ALL SECURITY OK")
