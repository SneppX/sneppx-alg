"""Ed25519 interop tests — C (via pybind11) vs PyNaCl/cryptography.

Operates as: @sneppx-alg-python

These tests verify that the SNEPPX C Ed25519 implementation produces
signatures identical to PyNaCl and is fully cross-verifiable.
They also verify the SHA-512 fix (RFC 8032 compliant output) that underpins
Ed25519.
"""

import hashlib
import importlib.util
import os
import sys

import pytest

# ── Load the _SNEPPX_c extension directly (bypasses heavy __init__) ──────
_BINDINGS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "bindings", "python", "SneppX_ALG"
)
_PYD_PATH = os.path.join(_BINDINGS_DIR, "_SNEPPX_c.cp311-win_amd64.pyd")
if os.path.exists(_PYD_PATH):
    _spec = importlib.util.spec_from_file_location("_SNEPPX_c", _PYD_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    crypto = _mod.crypto
    HAS_C = True
else:
    crypto = None
    HAS_C = False

# RFC 8032 test vector seed
RFC8032_SEED = bytes.fromhex(
    "9d61b19deffd5a60ba844af492ec2cc4"
    "4449c5697b3a6339cb3cc4cd8a14442a"
)

# L — Ed25519 group order
_L_BYTES = bytes([
    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58,
    0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10,
])
ED25519_L = int.from_bytes(_L_BYTES, "little")

pytestmark = pytest.mark.skipif(not HAS_C, reason="C extension _SNEPPX_c not available")


# ── SHA-512 tests ────────────────────────────────────────────────────────

@pytest.mark.parametrize("data", [
    b"",
    b"abc",
    b"hello world",
    b"The quick brown fox jumps over the lazy dog",
    b"a" * 1024,
])
def test_sha512_matches_hashlib(data):
    """C SHA-512 must match Python's hashlib."""
    c_result = crypto.sha512(data)
    py_result = hashlib.sha512(data).digest()
    assert c_result == py_result


def test_sha512_rfc_abc():
    """SHA-512('abc') must match the RFC 8032 / NIST known answer."""
    expected = hashlib.sha512(b"abc").digest()
    assert crypto.sha512(b"abc") == expected


# ── Ed25519 interop tests ────────────────────────────────────────────────

def _build_sk64(seed: bytes, pk: bytes) -> bytes:
    """Build the 64-byte private key format: seed[32] || pk[32]."""
    return seed + pk


def test_ed25519_signature_matches_pynacl():
    """C ed25519_sign must produce identical signatures to PyNaCl."""
    from nacl.signing import SigningKey

    pynacl_sk = SigningKey(RFC8032_SEED)
    pk = bytes(pynacl_sk.verify_key)
    sk_64 = _build_sk64(RFC8032_SEED, pk)

    for msg in [b"", b"hello", b"test message for ed25519 interop"]:
        c_sig = crypto.ed25519_sign(msg, sk_64)
        pynacl_sig = pynacl_sk.sign(msg).signature
        assert c_sig == pynacl_sig, f"Signatures differ for message: {msg!r}"


def test_ed25519_pynacl_verifies_c_signature():
    """PyNaCl must accept signatures produced by the C implementation."""
    from nacl.signing import SigningKey
    from nacl.exceptions import BadSignatureError

    pynacl_sk = SigningKey(RFC8032_SEED)
    pk = bytes(pynacl_sk.verify_key)
    sk_64 = _build_sk64(RFC8032_SEED, pk)

    for msg in [b"", b"hello", b"test"]:
        c_sig = crypto.ed25519_sign(msg, sk_64)
        # PyNaCl verify expects signed_message = sig + msg
        try:
            pynacl_sk.verify_key.verify(c_sig + msg)
            ok = True
        except BadSignatureError:
            ok = False
        assert ok, f"PyNaCl could not verify C signature for: {msg!r}"


def test_ed25519_c_verifies_pynacl_signature():
    """C ed25519_verify must accept signatures produced by PyNaCl."""
    from nacl.signing import SigningKey

    pynacl_sk = SigningKey(RFC8032_SEED)
    pk = bytes(pynacl_sk.verify_key)
    sk_64 = _build_sk64(RFC8032_SEED, pk)

    for msg in [b"", b"hello", b"test"]:
        pynacl_sig = pynacl_sk.sign(msg).signature
        ret = crypto.ed25519_verify(pynacl_sig, msg, pk)
        assert ret == 1, f"C could not verify PyNaCl signature for: {msg!r}"


def test_ed25519_c_sign_verify_roundtrip():
    """C sign → C verify round-trip with deterministically derived keypair."""
    from nacl.signing import SigningKey

    pynacl_sk = SigningKey(RFC8032_SEED)
    pk = bytes(pynacl_sk.verify_key)
    sk_64 = _build_sk64(RFC8032_SEED, pk)

    for msg in [b"roundtrip", b"", b"longer message for testing"]:
        sig = crypto.ed25519_sign(msg, sk_64)
        assert crypto.ed25519_verify(sig, msg, pk) == 1


def test_ed25519_wrong_message_rejected():
    """C verify must reject signatures for wrong messages."""
    from nacl.signing import SigningKey

    pynacl_sk = SigningKey(RFC8032_SEED)
    pk = bytes(pynacl_sk.verify_key)
    sk_64 = _build_sk64(RFC8032_SEED, pk)

    msg = b"original"
    sig = crypto.ed25519_sign(msg, sk_64)
    ret = crypto.ed25519_verify(sig, b"tampered", pk)
    assert ret == 0


def test_ed25519_cryptography_interop():
    """C signatures must be verifiable with the `cryptography` library."""
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.exceptions import InvalidSignature

    pynacl_sk = ed25519.Ed25519PrivateKey.from_private_bytes(RFC8032_SEED)
    pk = pynacl_sk.public_key().public_bytes_raw()
    sk_64 = _build_sk64(RFC8032_SEED, pk)

    msg = b"cryptography interop test"
    c_sig = crypto.ed25519_sign(msg, sk_64)

    pub = ed25519.Ed25519PublicKey.from_public_bytes(pk)
    try:
        pub.verify(c_sig, msg)
        ok = True
    except InvalidSignature:
        ok = False
    assert ok, "cryptography library rejected C-generated signature"


def test_ed25519_cryptography_sign_c_verify():
    """`cryptography` signatures must be verifiable by C verify."""
    from cryptography.hazmat.primitives.asymmetric import ed25519

    priv = ed25519.Ed25519PrivateKey.from_private_bytes(RFC8032_SEED)
    pk = priv.public_key().public_bytes_raw()
    msg = b"cryptography signs, C verifies"
    py_sig = priv.sign(msg)

    ret = crypto.ed25519_verify(py_sig, msg, pk)
    assert ret == 1, "C could not verify cryptography signature"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
