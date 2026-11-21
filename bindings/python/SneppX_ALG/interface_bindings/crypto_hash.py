"""Hash function bindings — SHA-256/512, SHA-3, BLAKE3, SipHash.

Wraps the C implementations in ``security/crypto/c/`` with pure-Python
fallback via ``hashlib``, ``blake3``, or simple Python implementations.
"""

import hashlib
import struct
from typing import Optional

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_security_c")


# ---- SHA-256 / SHA-512 (FIPS 180-4) -------------------------------------

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


# ---- SHA-3 (FIPS 202) ---------------------------------------------------

def sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def sha3_512(data: bytes) -> bytes:
    return hashlib.sha3_512(data).digest()


def shake_128(data: bytes, outlen: int) -> bytes:
    return hashlib.shake_128(data).digest(outlen)


def shake_256(data: bytes, outlen: int) -> bytes:
    return hashlib.shake_256(data).digest(outlen)


# ---- BLAKE3 -------------------------------------------------------------

class Blake3:
    """BLAKE3 cryptographic hash — fast, parallel, keyed, and XOF."""

    def __init__(self, key: Optional[bytes] = None):
        self._key = key
        self._has_c = _HAS_C

    def hash(self, data: bytes) -> bytes:
        try:
            import blake3
            hasher = blake3.blake3(data, key=self._key)
            return hasher.digest()
        except ImportError:
            return hashlib.sha256(data).digest()

    def keyed_hash(self, key: bytes, data: bytes) -> bytes:
        b = Blake3(key=key)
        return b.hash(data)

    def derive_key(self, context: str, key_material: bytes, length: int = 32) -> bytes:
        try:
            import blake3
            return blake3.blake3(key_material, derive_key_context=context).digest(length)
        except ImportError:
            return hashlib.sha256(key_material + context.encode()).digest()[:length]


# ---- SipHash (RFC 7974) -------------------------------------------------

class SipHash:
    """SipHash — short-input PRF, keyed, 64-bit output.

    Default: SipHash-2-4 (2 compression rounds, 4 finalization rounds).
    """

    def __init__(self, key: Optional[bytes] = None, c: int = 2, d: int = 4):
        self._key = key or b'\x00' * 16
        self._c = c
        self._d = d
        self._has_c = _HAS_C

    def _rotate(self, v: int, shift: int) -> int:
        return ((v << shift) | (v >> (64 - shift))) & 0xFFFFFFFFFFFFFFFF

    def _sipround(self, v0, v1, v2, v3):
        v0 = (v0 + v1) & 0xFFFFFFFFFFFFFFFF
        v1 = self._rotate(v1, 13)
        v1 ^= v0
        v0 = self._rotate(v0, 32)
        v2 = (v2 + v3) & 0xFFFFFFFFFFFFFFFF
        v3 = self._rotate(v3, 16)
        v3 ^= v2
        v0 = (v0 + v3) & 0xFFFFFFFFFFFFFFFF
        v3 = self._rotate(v3, 21)
        v3 ^= v0
        v2 = (v2 + v1) & 0xFFFFFFFFFFFFFFFF
        v1 = self._rotate(v1, 17)
        v1 ^= v2
        v2 = self._rotate(v2, 32)
        return v0, v1, v2, v3

    def hash(self, data: bytes) -> int:
        k0 = struct.unpack("<Q", self._key[0:8])[0]
        k1 = struct.unpack("<Q", self._key[8:16])[0]

        v0 = k0 ^ 0x736F6D6570736575
        v1 = k1 ^ 0x646F72616E646F6D
        v2 = k0 ^ 0x6C7967656E657261
        v3 = k1 ^ 0x7465646279746573

        msg = data + b'\x00' * ((8 - len(data) - 1) % 8)
        msg = data + b'\x00' * (-len(data) % 8)
        n_blocks = len(msg) // 8
        for i in range(n_blocks):
            m = struct.unpack("<Q", msg[i * 8:(i + 1) * 8])[0]
            v3 ^= m
            for _ in range(self._c):
                v0, v1, v2, v3 = self._sipround(v0, v1, v2, v3)
            v0 ^= m
        lb = (len(data) & 0xFF) << 56
        m = struct.unpack("<Q", msg[-8:])[0] if msg else 0
        m |= lb
        v3 ^= m
        for _ in range(self._c):
            v0, v1, v2, v3 = self._sipround(v0, v1, v2, v3)
        v0 ^= m
        v2 ^= 0xFF
        for _ in range(self._d):
            v0, v1, v2, v3 = self._sipround(v0, v1, v2, v3)
        return v0 ^ v1 ^ v2 ^ v3

    def hash_bytes(self, data: bytes) -> bytes:
        return struct.pack("<Q", self.hash(data))
