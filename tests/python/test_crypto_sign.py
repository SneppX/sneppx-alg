"""Tests for crypto_sign.py — Ed25519, Dilithium, SPHINCS+."""

import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

import SneppX_ALG.interface_bindings.crypto_sign as crypto_sign_module
from SneppX_ALG.interface_bindings import Ed25519, Dilithium, SphincsPlus


def test_ed25519_keypair():
    sk, pk = Ed25519.keypair()
    assert len(sk) in (32, 64)
    assert len(pk) == 32


def test_ed25519_sign_verify():
    sk, pk = Ed25519.keypair()
    msg = b"hello world"
    sig = Ed25519.sign(sk, msg)
    assert Ed25519.verify(pk, msg, sig)


def test_ed25519_wrong_message():
    sk, pk = Ed25519.keypair()
    msg = b"hello"
    sig = Ed25519.sign(sk, msg)
    assert not Ed25519.verify(pk, b"wrong", sig)


def test_ed25519_seed():
    seed = b"s" * 32
    sk1, pk1 = Ed25519.keypair(seed)
    sk2, pk2 = Ed25519.keypair(seed)
    assert sk1 == sk2
    assert pk1 == pk2


def test_ed25519_empty_message():
    sk, pk = Ed25519.keypair()
    sig = Ed25519.sign(sk, b"")
    assert Ed25519.verify(pk, b"", sig)


@pytest.mark.skipif(crypto_sign_module._C is None, reason="native post-quantum backend unavailable")
def test_dilithium_keypair():
    d = Dilithium(Dilithium.MODE_ML_DSA_44)
    pk, sk = d.keypair()
    assert len(pk) > 0
    assert len(sk) > 0


@pytest.mark.skipif(crypto_sign_module._C is None, reason="native post-quantum backend unavailable")
def test_dilithium_modes():
    for mode in [Dilithium.MODE_ML_DSA_44, Dilithium.MODE_ML_DSA_65, Dilithium.MODE_ML_DSA_87]:
        d = Dilithium(mode)
        pk, sk = d.keypair()
        sig = d.sign(sk, b"test")
        assert d.verify(pk, b"test", sig)


@pytest.mark.skipif(crypto_sign_module._C is None, reason="native post-quantum backend unavailable")
def test_sphincsplus_keypair():
    s = SphincsPlus("128s")
    pk, sk = s.keypair()
    assert len(pk) == 32
    assert len(sk) == 64


@pytest.mark.skipif(crypto_sign_module._C is None, reason="native post-quantum backend unavailable")
def test_sphincsplus_variants():
    for v in ["128s", "128f", "192s", "192f", "256s", "256f"]:
        s = SphincsPlus(v)
        pk, sk = s.keypair()
        sig = s.sign(sk, b"msg")
        assert s.verify(pk, b"msg", sig)


@pytest.mark.skipif(crypto_sign_module._C is None, reason="native post-quantum backend unavailable")
def test_dilithium_sign_verify():
    d = Dilithium()
    pk, sk = d.keypair()
    msg = b"test message for dilithium"
    sig = d.sign(sk, msg)
    assert d.verify(pk, msg, sig)


def test_post_quantum_fallback_fails_closed():
    if crypto_sign_module._C is not None:
        pytest.skip("native post-quantum backend is available")

    dilithium = Dilithium()
    with pytest.raises(RuntimeError, match="native SneppX crypto extension"):
        dilithium.keypair()
    with pytest.raises(RuntimeError, match="native SneppX crypto extension"):
        dilithium.sign(b"placeholder-secret", b"message")
    assert dilithium.verify(b"placeholder-public", b"message", b"signature") is False

    sphincs = SphincsPlus()
    with pytest.raises(RuntimeError, match="native SneppX crypto extension"):
        sphincs.keypair()
    with pytest.raises(RuntimeError, match="native SneppX crypto extension"):
        sphincs.sign(b"placeholder-secret", b"message")
    assert sphincs.verify(b"placeholder-public", b"message", b"signature") is False


if __name__ == "__main__":
    test_ed25519_keypair()
    test_ed25519_sign_verify()
    test_ed25519_wrong_message()
    test_ed25519_seed()
    test_ed25519_empty_message()
    if crypto_sign_module._C is not None:
        test_dilithium_keypair()
        test_dilithium_modes()
        test_sphincsplus_keypair()
        test_sphincsplus_variants()
        test_dilithium_sign_verify()
    else:
        test_post_quantum_fallback_fails_closed()
    print("All crypto_sign tests passed.")
