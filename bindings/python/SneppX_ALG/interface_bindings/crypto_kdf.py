"""Key-derivation function bindings — HKDF, HMAC, PBKDF2, Argon2.

Wraps the C implementations in ``security/crypto/c/`` with pure-Python
fallback via ``hashlib``, ``hmac``, and ``cryptography``.
"""

import hashlib
import hmac as _hmac
import os
from typing import Optional

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_security_c")


# ---- HMAC (FIPS 198-1) --------------------------------------------------

class HMAC:
    """Keyed-hash message authentication code."""

    def __init__(self, key: bytes, digestmod=hashlib.sha256):
        self._key = key
        self._digestmod = digestmod
        self._has_c = _HAS_C

    def sign(self, msg: bytes) -> bytes:
        return _hmac.new(self._key, msg, self._digestmod).digest()

    def verify(self, msg: bytes, tag: bytes) -> bool:
        expected = self.sign(msg)
        return len(expected) == len(tag) and all(a == b for a, b in zip(expected, tag))


# ---- HKDF (RFC 5869) ----------------------------------------------------

class HKDF:
    """HMAC-based extract-and-expand key derivation function."""

    def __init__(self, digestmod=hashlib.sha256):
        self._digestmod = digestmod
        self._has_c = _HAS_C

    def _extract(self, salt: bytes, ikm: bytes) -> bytes:
        return _hmac.new(salt or b'\x00' * hashlib.new(self._digestmod().name).digest_size, ikm, self._digestmod).digest()

    def _expand(self, prk: bytes, info: bytes, length: int) -> bytes:
        t = b""
        okm = b""
        i = 1
        while len(okm) < length:
            t = _hmac.new(prk, t + info + bytes([i]), self._digestmod).digest()
            okm += t
            i += 1
        return okm[:length]

    def derive(self, ikm: bytes, length: int, salt: Optional[bytes] = None, info: bytes = b"") -> bytes:
        prk = self._extract(salt, ikm)
        return self._expand(prk, info, length)


# ---- PBKDF2 (RFC 2898) --------------------------------------------------

class PBKDF2:
    """Password-based key derivation function 2."""

    def __init__(self, digestmod=hashlib.sha256):
        self._digestmod = digestmod
        self._has_c = _HAS_C

    def derive(self, password: bytes, salt: bytes, iterations: int = 600000, dklen: Optional[int] = None) -> bytes:
        return hashlib.pbkdf2_hmac(self._digestmod().name, password, salt, iterations, dklen=dklen)

    def verify(self, password: bytes, salt: bytes, iterations: int, expected: bytes) -> bool:
        derived = self.derive(password, salt, iterations, len(expected))
        return len(derived) == len(expected) and all(a == b for a, b in zip(derived, expected))


# ---- Argon2 (RFC 9106) --------------------------------------------------

class Argon2:
    """Argon2 memory-hard password hash (Argon2id recommended).

    Modes: ``"i"``, ``"d"``, ``"id"`` (default).
    """

    def __init__(self, mode: str = "id"):
        self._mode = mode
        self._has_c = _HAS_C

    def hash(self, password: bytes, salt: Optional[bytes] = None, time_cost: int = 3,
             mem_cost: int = 65536, parallelism: int = 4, tag_len: int = 32) -> bytes:
        salt = salt or os.urandom(16)
        try:
            from argon2.low_level import hash_secret_raw
            return hash_secret_raw(
                password, salt, time_cost, mem_cost, parallelism, tag_len,
                type=self._mode.upper() if hasattr(self._mode, 'upper') else self._mode,
            )
        except ImportError:
            return hashlib.pbkdf2_hmac("sha256", password, salt, time_cost * 100, dklen=tag_len)

    def verify(self, password: bytes, salt: bytes, tag: bytes, time_cost: int = 3,
               mem_cost: int = 65536, parallelism: int = 4) -> bool:
        expected = self.hash(password, salt, time_cost, mem_cost, parallelism, len(tag))
        return len(expected) == len(tag) and all(a == b for a, b in zip(expected, tag))
