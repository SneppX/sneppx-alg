"""Tests for crypto_symmetric.py — AESGCM, ChaCha20, Poly1305, AEAD."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import AESGCM, ChaCha20, Poly1305, AEAD


def test_aesgcm_encrypt_decrypt():
    aes = AESGCM(b"k" * 32)
    pt = b"hello world"
    ct = aes.encrypt(pt)
    dec = aes.decrypt(ct)
    assert dec == pt


def test_aesgcm_with_aad():
    aes = AESGCM(b"k" * 32)
    pt = b"data with aad"
    aad = b"associated data"
    ct = aes.encrypt(pt, aad=aad)
    dec = aes.decrypt(ct, aad=aad)
    assert dec == pt


def test_aesgcm_empty():
    aes = AESGCM(b"k" * 32)
    ct = aes.encrypt(b"")
    dec = aes.decrypt(ct)
    assert dec == b""


def test_aesgcm_key_sizes():
    for ks in [16, 24, 32]:
        aes = AESGCM(b"k" * ks)
        ct = aes.encrypt(b"test")
        dec = aes.decrypt(ct)
        assert dec == b"test"


def test_aesgcm_invalid_key():
    try:
        AESGCM(b"short")
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_chacha20_encrypt_decrypt():
    chacha = ChaCha20(b"k" * 32)
    pt = b"hello chacha"
    ct = chacha.encrypt(pt)
    dec = chacha.decrypt(ct)
    assert dec == pt


def test_chacha20_empty():
    chacha = ChaCha20(b"k" * 32)
    ct = chacha.encrypt(b"")
    dec = chacha.decrypt(ct)
    assert dec == b""


def test_chacha20_invalid_key():
    try:
        ChaCha20(b"short")
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_poly1305_mac_verify():
    poly = Poly1305(b"k" * 32)
    tag = poly.mac(b"test message")
    assert poly.verify(b"test message", tag)


def test_poly1305_wrong_key():
    poly1 = Poly1305(b"k" * 32)
    poly2 = Poly1305(b"x" * 32)
    tag = poly1.mac(b"msg")
    assert not poly2.verify(b"msg", tag)


def test_aead_aes():
    aead = AEAD(b"k" * 32, "aes-256-gcm")
    ct = aead.encrypt(b"data")
    dec = aead.decrypt(ct)
    assert dec == b"data"


def test_aead_chacha():
    aead = AEAD(b"k" * 32, "chacha20")
    ct = aead.encrypt(b"data")
    dec = aead.decrypt(ct)
    assert dec == b"data"


def test_aead_with_aad():
    aead = AEAD(b"k" * 32)
    ct = aead.encrypt(b"secret", aad=b"header")
    dec = aead.decrypt(ct, aad=b"header")
    assert dec == b"secret"


if __name__ == "__main__":
    test_aesgcm_encrypt_decrypt()
    test_aesgcm_with_aad()
    test_aesgcm_empty()
    test_aesgcm_key_sizes()
    test_aesgcm_invalid_key()
    test_chacha20_encrypt_decrypt()
    test_chacha20_empty()
    test_chacha20_invalid_key()
    test_poly1305_mac_verify()
    test_poly1305_wrong_key()
    test_aead_aes()
    test_aead_chacha()
    test_aead_with_aad()
    print("All crypto_symmetric tests passed.")
