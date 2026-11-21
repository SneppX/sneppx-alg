"""Tests for crypto_kem.py — KyberKEM, X25519."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import KyberKEM, X25519


def test_kyber_keypair():
    k = KyberKEM(KyberKEM.MODE_ML_KEM_512)
    pk, sk = k.keypair()
    assert len(pk) > 0
    assert len(sk) > 0


def test_kyber_encaps_decaps():
    k = KyberKEM()
    pk, sk = k.keypair()
    ct, ss = k.encapsulate(pk)
    assert len(ct) > 0
    assert len(ss) == 32
    ss2 = k.decapsulate(sk, ct)
    assert ss2 is not None


def test_kyber_modes():
    for mode in [KyberKEM.MODE_ML_KEM_512, KyberKEM.MODE_ML_KEM_768, KyberKEM.MODE_ML_KEM_1024]:
        k = KyberKEM(mode)
        pk, sk = k.keypair()
        assert len(pk) > 0
        ct, ss = k.encapsulate(pk)
        ss2 = k.decapsulate(sk, ct)
        assert ss2 is not None


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
    assert len(ss_b) == 32


def test_x25519_scalar_mult():
    sk, pk = X25519.keypair()
    result = X25519.scalar_mult(sk, pk)
    assert len(result) == 32


if __name__ == "__main__":
    test_kyber_keypair()
    test_kyber_encaps_decaps()
    test_kyber_modes()
    test_x25519_keypair()
    test_x25519_keypair_seed()
    test_x25519_dh()
    test_x25519_scalar_mult()
    print("All crypto_kem tests passed.")
