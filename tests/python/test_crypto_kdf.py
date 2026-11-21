"""Tests for crypto_kdf.py — HMAC, HKDF, PBKDF2, Argon2."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import HMAC_, HKDF, PBKDF2, Argon2


def test_hmac_sign():
    hm = HMAC_(b"key")
    tag = hm.sign(b"message")
    assert len(tag) == 32
    assert hm.verify(b"message", tag)


def test_hmac_wrong_key():
    hm1 = HMAC_(b"key1")
    hm2 = HMAC_(b"key2")
    tag = hm1.sign(b"msg")
    assert not hm2.verify(b"msg", tag)


def test_hmac_wrong_message():
    hm = HMAC_(b"key")
    tag = hm.sign(b"msg1")
    assert not hm.verify(b"msg2", tag)


def test_hmac_empty():
    hm = HMAC_(b"key")
    tag = hm.sign(b"")
    assert hm.verify(b"", tag)


def test_hmac_sha512():
    import hashlib
    hm = HMAC_(b"key", digestmod=hashlib.sha512)
    tag = hm.sign(b"msg")
    assert len(tag) == 64


def test_hkdf_derive():
    hkdf = HKDF()
    okm = hkdf.derive(b"ikm", 32, salt=b"salt", info=b"info")
    assert len(okm) == 32


def test_hkdf_no_salt():
    hkdf = HKDF()
    okm = hkdf.derive(b"ikm", 16)
    assert len(okm) == 16


def test_hkdf_deterministic():
    hkdf = HKDF()
    a = hkdf.derive(b"ikm", 16, salt=b"s")
    b = hkdf.derive(b"ikm", 16, salt=b"s")
    assert a == b


def test_hkdf_varying_length():
    hkdf = HKDF()
    for length in [1, 16, 32, 64, 128]:
        okm = hkdf.derive(b"ikm", length, salt=b"s")
        assert len(okm) == length


def test_pbkdf2_derive():
    p = PBKDF2()
    dk = p.derive(b"password", b"salt", iterations=100, dklen=16)
    assert len(dk) == 16


def test_pbkdf2_verify():
    p = PBKDF2()
    dk = p.derive(b"pass", b"salt", iterations=100, dklen=16)
    assert p.verify(b"pass", b"salt", 100, dk)


def test_pbkdf2_wrong_password():
    p = PBKDF2()
    dk = p.derive(b"pass1", b"salt", 100, 16)
    assert not p.verify(b"pass2", b"salt", 100, dk)


def test_pbkdf2_deterministic():
    p = PBKDF2()
    a = p.derive(b"p", b"s", 100, 16)
    b = p.derive(b"p", b"s", 100, 16)
    assert a == b


def test_argon2_hash():
    a2 = Argon2("id")
    h = a2.hash(b"password", b"salt123456789012", time_cost=1, mem_cost=1024)
    assert len(h) == 32


def test_argon2_verify():
    a2 = Argon2("id")
    h = a2.hash(b"pass", b"salt123456789012", time_cost=1, mem_cost=1024)
    assert a2.verify(b"pass", b"salt123456789012", h, time_cost=1, mem_cost=1024)


def test_argon2_deterministic():
    a2 = Argon2("id")
    a = a2.hash(b"p", b"salt123456789012", time_cost=1, mem_cost=1024)
    b = a2.hash(b"p", b"salt123456789012", time_cost=1, mem_cost=1024)
    assert a == b


if __name__ == "__main__":
    test_hmac_sign()
    test_hmac_wrong_key()
    test_hmac_wrong_message()
    test_hmac_empty()
    test_hmac_sha512()
    test_hkdf_derive()
    test_hkdf_no_salt()
    test_hkdf_deterministic()
    test_hkdf_varying_length()
    test_pbkdf2_derive()
    test_pbkdf2_verify()
    test_pbkdf2_wrong_password()
    test_pbkdf2_deterministic()
    test_argon2_hash()
    test_argon2_verify()
    test_argon2_deterministic()
    print("All crypto_kdf tests passed.")
