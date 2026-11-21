"""Tests for secure_updates.py — S7 secure updates.

Signed updates use Ed25519 over BLAKE3. Verification is fail-closed: a
trusted public key must be registered and the signature must verify,
otherwise ``apply`` rejects the update.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import SignedUpdate, Ed25519, Blake3


def _keypair():
    return Ed25519.keypair()


def _metadata_payload(metadata):
    return json.dumps(metadata, sort_keys=True).encode()


def test_hash_package_returns_32_bytes():
    h = SignedUpdate.hash_package(b"test data")
    assert isinstance(h, (bytes, bytearray))
    assert len(h) == 32


def test_hash_package_deterministic():
    assert SignedUpdate.hash_package(b"x") == SignedUpdate.hash_package(b"x")


def test_hash_package_sensitive_to_input():
    assert SignedUpdate.hash_package(b"a") != SignedUpdate.hash_package(b"b")


def test_sign_verify_roundtrip():
    sk, pk = _keypair()
    data, sig = SignedUpdate.sign_update(b"important update", sk)
    assert SignedUpdate.verify_signature(b"important update", sig, pk)


def test_sign_verify_uses_blake3_digest():
    sk, pk = _keypair()
    _, sig = SignedUpdate.sign_update(b"data", sk)
    # Signature is over BLAKE3(data), so it verifies against the digest...
    digest = Blake3().hash(b"data")
    assert Ed25519.verify(pk, digest, sig)
    # ...but not against the raw message bytes.
    assert not Ed25519.verify(pk, b"data", sig)


def test_verify_wrong_key():
    sk, pk = _keypair()
    _, wrong_pk = _keypair()
    data, sig = SignedUpdate.sign_update(b"data", sk)
    assert not SignedUpdate.verify_signature(b"data", sig, wrong_pk)


def test_verify_tampered_data():
    sk, pk = _keypair()
    _, sig = SignedUpdate.sign_update(b"original", sk)
    assert not SignedUpdate.verify_signature(b"tampered", sig, pk)


def test_register_update_pending():
    su = SignedUpdate()
    su.register_update("v2", {"version": "2.0", "size": 1234})
    assert su.status("v2") == "pending"


def test_apply_without_signature_rejected():
    su = SignedUpdate()
    su.register_update("v1", {})
    # No signature on file and no trusted key -> fail closed.
    assert not su.apply("v1")
    assert su.status("v1") == "pending"


def test_apply_with_valid_signature():
    sk, pk = _keypair()
    su = SignedUpdate()
    su.set_trusted_public_key(pk)
    metadata = {"version": "2.0", "size": 1234}
    payload = _metadata_payload(metadata)
    _, sig = SignedUpdate.sign_update(payload, sk)
    su.register_update("v2", metadata, sig)
    assert su.apply("v2")
    assert su.status("v2") == "applied"


def test_apply_fail_closed_no_trusted_key():
    sk, pk = _keypair()
    su = SignedUpdate()
    # Trusted key deliberately NOT registered.
    metadata = {"version": "2.0"}
    payload = _metadata_payload(metadata)
    _, sig = SignedUpdate.sign_update(payload, sk)
    su.register_update("v2", metadata, sig)
    assert not su.apply("v2")
    assert su.status("v2") == "pending"


def test_apply_wrong_signature_rejected():
    sk, pk = _keypair()
    su = SignedUpdate()
    su.set_trusted_public_key(pk)
    metadata = {"version": "2.0"}
    payload = _metadata_payload(metadata)
    _, sig = SignedUpdate.sign_update(payload, sk)
    # Tamper the signature bytes.
    bad_sig = bytes(sig)
    bad_sig = bad_sig[:10] + bytes([bad_sig[10] ^ 0xFF]) + bad_sig[11:]
    su.register_update("v2", metadata, bad_sig)
    assert not su.apply("v2")
    assert su.status("v2") == "pending"


def test_apply_idempotent_second_call_rejected():
    sk, pk = _keypair()
    su = SignedUpdate()
    su.set_trusted_public_key(pk)
    metadata = {"version": "2.0"}
    payload = _metadata_payload(metadata)
    _, sig = SignedUpdate.sign_update(payload, sk)
    su.register_update("v1", metadata, sig)
    assert su.apply("v1")
    assert not su.apply("v1")
    assert su.status("v1") == "applied"


def test_apply_missing_update():
    su = SignedUpdate()
    su.set_trusted_public_key(b"\x01" * 32)
    assert not su.apply("missing")
    assert su.status("missing") is None


def test_rollback():
    sk, pk = _keypair()
    su = SignedUpdate()
    su.set_trusted_public_key(pk)
    metadata = {"version": "1.0"}
    payload = _metadata_payload(metadata)
    _, sig = SignedUpdate.sign_update(payload, sk)
    su.register_update("v1", metadata, sig)
    su.apply("v1")
    assert su.rollback("v1")
    assert su.status("v1") == "rolled_back"


def test_rollback_missing():
    su = SignedUpdate()
    assert not su.rollback("missing")


def test_container_sbom():
    cs = SignedUpdate.__new__(type("CS", (), {}))  # noqa
    from SneppX_ALG.interface_bindings.secure_updates import ContainerSecurity
    cs = ContainerSecurity()
    sbom = cs.generate_sbom([
        {"name": "numpy", "version": "1.26.0"},
        {"name": "requests", "version": "2.31.0"},
    ])
    assert sbom["bomFormat"] == "CycloneDX"
    assert len(sbom["components"]) == 2


def test_container_sbom_register():
    from SneppX_ALG.interface_bindings.secure_updates import ContainerSecurity
    cs = ContainerSecurity()
    sbom = cs.generate_sbom([{"name": "pkg", "version": "1.0"}])
    cs.register_sbom("image:latest", sbom)
    assert cs.scan_cves("image:latest") == []


def test_container_vulnerability():
    from SneppX_ALG.interface_bindings.secure_updates import ContainerSecurity
    cs = ContainerSecurity()
    cs.add_vulnerability("img:v1", {"id": "CVE-2024-0001", "severity": "critical"})
    vulns = cs.scan_cves("img:v1")
    assert len(vulns) == 1
    assert vulns[0]["id"] == "CVE-2024-0001"


def test_container_compliance():
    from SneppX_ALG.interface_bindings.secure_updates import ContainerSecurity
    cs = ContainerSecurity()
    cs.add_vulnerability("img:v2", {"id": "CVE-1", "severity": "critical"})
    assert not cs.is_compliant("img:v2", {"max_critical": 0})
    assert cs.is_compliant("img:v2", {"max_critical": 1})


if __name__ == "__main__":
    test_hash_package_returns_32_bytes()
    test_hash_package_deterministic()
    test_hash_package_sensitive_to_input()
    test_sign_verify_roundtrip()
    test_sign_verify_uses_blake3_digest()
    test_verify_wrong_key()
    test_verify_tampered_data()
    test_register_update_pending()
    test_apply_without_signature_rejected()
    test_apply_with_valid_signature()
    test_apply_fail_closed_no_trusted_key()
    test_apply_wrong_signature_rejected()
    test_apply_idempotent_second_call_rejected()
    test_apply_missing_update()
    test_rollback()
    test_rollback_missing()
    test_container_sbom()
    test_container_sbom_register()
    test_container_vulnerability()
    test_container_compliance()
    print("All secure_updates tests passed.")
