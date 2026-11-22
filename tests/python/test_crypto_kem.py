"""Tests for crypto_kem.py — KyberKEM, X25519.

KyberKEM must fail closed: the Python bindings have no validated ML-KEM
backend, and fabricating random secrets used to let two peers "succeed"
while deriving unrelated keys.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python")
)

from SneppX_ALG.interface_bindings import KyberKEM, X25519


def test_kyber_keypair_fails_closed():
    k = KyberKEM(KyberKEM.MODE_ML_KEM_512)
    with pytest.raises(RuntimeError, match="ML-KEM"):
        k.keypair()


def test_kyber_encaps_decaps_fails_closed():
    k = KyberKEM()
    with pytest.raises(RuntimeError, match="ML-KEM"):
        k.encapsulate(b"\x00" * 800)
    with pytest.raises(RuntimeError, match="ML-KEM"):
        k.decapsulate(b"\x00" * 1632, b"\x00" * 768)


def test_kyber_modes_fail_closed():
    for mode in [
        KyberKEM.MODE_ML_KEM_512,
        KyberKEM.MODE_ML_KEM_768,
        KyberKEM.MODE_ML_KEM_1024,
    ]:
        k = KyberKEM(mode)
        with pytest.raises(RuntimeError, match="ML-KEM"):
            k.keypair()


def test_x25519_keypair():
    sk, pk = X25519.keypair()
    assert len(sk) == 32
    assert len(pk) == 32


def test_x25519_keypair_seed():
    priv = b"p" * 32
    sk, pk = X25519.keypair(private=priv)
    assert sk == priv


def test_x25519_dh():
    sk_a, pk_a = X25519.keypair()
    sk_b, pk_b = X25519.keypair()
    ss_a = X25519.dh(sk_a, pk_b)
    ss_b = X25519.dh(sk_b, pk_a)
    assert len(ss_a) == 32
    assert ss_a == ss_b


def test_x25519_scalar_mult():
    sk, pk = X25519.keypair()
    result = X25519.scalar_mult(sk, pk)
    assert len(result) == 32


def test_x25519_fails_closed_without_backend(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "cryptography.hazmat.primitives.asymmetric.x25519", None
    )
    with pytest.raises(RuntimeError, match="cryptography"):
        X25519.keypair()
    with pytest.raises(RuntimeError, match="cryptography"):
        X25519.dh(b"\x00" * 32, b"\x00" * 32)
