"""Regression: AEAD bindings must fail closed when the crypto backend is absent."""

import sys

import pytest

from SneppX_ALG.interface_bindings import crypto_symmetric as cs


@pytest.mark.parametrize(
    "factory,key",
    [
        (cs.AESGCM, b"A" * 32),
        (cs.ChaCha20, b"B" * 32),
    ],
)
def test_encrypt_refuses_without_backend(monkeypatch, factory, key):
    monkeypatch.setitem(sys.modules, "cryptography", None)
    cipher = factory(key)
    with pytest.raises(RuntimeError, match="cryptography"):
        cipher.encrypt(b"top secret")


@pytest.mark.parametrize(
    "factory,key",
    [
        (cs.AESGCM, b"A" * 32),
        (cs.ChaCha20, b"B" * 32),
    ],
)
def test_decrypt_refuses_without_backend(monkeypatch, factory, key):
    monkeypatch.setitem(sys.modules, "cryptography", None)
    cipher = factory(key)
    with pytest.raises(RuntimeError, match="cryptography"):
        cipher.decrypt(b"\x00" * 28)


def test_aead_roundtrip_with_real_backend():
    pytest.importorskip("cryptography")
    for cipher in (cs.AESGCM(b"A" * 32), cs.ChaCha20(b"B" * 32)):
        blob = cipher.encrypt(b"top secret", aad=b"header")
        assert cipher.decrypt(blob, aad=b"header") == b"top secret"
