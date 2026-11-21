"""Tests for crypto_hash.py — SHA-2/3, BLAKE3, SipHash."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import (
    sha256, sha512, sha3_256, sha3_512, shake_128, shake_256,
    Blake3, SipHash,
)


def test_sha256():
    h = sha256(b"hello")
    assert len(h) == 32
    assert sha256(b"hello") == sha256(b"hello")
    assert sha256(b"hello") != sha256(b"world")


def test_sha512():
    h = sha512(b"hello")
    assert len(h) == 64


def test_sha3_256():
    h = sha3_256(b"hello")
    assert len(h) == 32


def test_sha3_512():
    h = sha3_512(b"hello")
    assert len(h) == 64


def test_shake_128():
    h = shake_128(b"hello", 16)
    assert len(h) == 16
    h2 = shake_128(b"hello", 32)
    assert len(h2) == 32


def test_shake_256():
    h = shake_256(b"hello", 16)
    assert len(h) == 16


def test_deterministic():
    data = b"test data"
    assert sha256(data) == sha256(data)
    assert sha3_256(data) == sha3_256(data)
    assert shake_128(data, 16) == shake_128(data, 16)


def test_blake3_hash():
    b3 = Blake3()
    h = b3.hash(b"hello")
    assert len(h) == 32


def test_blake3_keyed():
    key = b"k" * 32
    h = Blake3(key=key).hash(b"data")
    assert len(h) == 32


def test_blake3_derive_key():
    b3 = Blake3()
    dk = b3.derive_key("test context", b"ikm", 16)
    assert len(dk) == 16


def test_blake3_deterministic():
    b3 = Blake3()
    assert b3.hash(b"x") == b3.hash(b"x")
    assert b3.hash(b"x") != b3.hash(b"y")


def test_siphash():
    sh = SipHash(b"k" * 16)
    h = sh.hash(b"data")
    assert isinstance(h, int)
    assert 0 <= h < 2**64


def test_siphash_bytes():
    sh = SipHash(b"k" * 16)
    h = sh.hash_bytes(b"data")
    assert len(h) == 8


def test_siphash_deterministic():
    sh = SipHash(b"k" * 16)
    assert sh.hash(b"x") == sh.hash(b"x")
    assert sh.hash_bytes(b"x") == sh.hash_bytes(b"x")


def test_siphash_different_keys():
    h1 = SipHash(b"k" * 16).hash(b"data")
    h2 = SipHash(b"x" * 16).hash(b"data")
    assert h1 != h2


def test_siphash_empty():
    sh = SipHash(b"k" * 16)
    h = sh.hash(b"")
    assert isinstance(h, int)


if __name__ == "__main__":
    test_sha256()
    test_sha512()
    test_sha3_256()
    test_sha3_512()
    test_shake_128()
    test_shake_256()
    test_deterministic()
    test_blake3_hash()
    test_blake3_keyed()
    test_blake3_derive_key()
    test_blake3_deterministic()
    test_siphash()
    test_siphash_bytes()
    test_siphash_deterministic()
    test_siphash_different_keys()
    test_siphash_empty()
    print("All crypto_hash tests passed.")
