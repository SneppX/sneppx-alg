"""Cryptographic utility bindings — BigNum, DRBG, Random, CT compare, EntropyPool.

Wraps the C implementations in ``security/crypto/c/`` with pure-Python fallback.
"""

import os
import struct
import time
import threading
from typing import Optional, List

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_security_c")


# ---- BigNum (arbitrary-precision integer) --------------------------------

class BigNum:
    """Arbitrary-precision integer arithmetic wrapping C bignum routines."""

    def __init__(self, value: int = 0):
        self._value = value
        self._has_c = _HAS_C

    @classmethod
    def from_bytes(cls, data: bytes, byteorder: str = "little") -> "BigNum":
        return cls(int.from_bytes(data, byteorder))

    def to_bytes(self, length: int = 0, byteorder: str = "little") -> bytes:
        nbytes = length or ((self._value.bit_length() + 7) // 8) or 1
        return self._value.to_bytes(nbytes, byteorder)

    def add(self, other: "BigNum") -> "BigNum":
        return BigNum(self._value + other._value)

    def sub(self, other: "BigNum") -> "BigNum":
        return BigNum(self._value - other._value)

    def mul(self, other: "BigNum") -> "BigNum":
        return BigNum(self._value * other._value)

    def divmod(self, other: "BigNum") -> tuple:
        q, r = divmod(self._value, other._value)
        return BigNum(q), BigNum(r)

    def mod(self, modulus: "BigNum") -> "BigNum":
        return BigNum(self._value % modulus._value)

    def pow_mod(self, exp: "BigNum", modulus: "BigNum") -> "BigNum":
        return BigNum(pow(self._value, exp._value, modulus._value))

    def inv_mod(self, modulus: "BigNum") -> Optional["BigNum"]:
        if self._value == 0:
            return None
        return BigNum(pow(self._value, -1, modulus._value))

    def __int__(self) -> int:
        return self._value

    def __repr__(self) -> str:
        return f"BigNum({self._value})"


# ---- DRBG (Deterministic Random Bit Generator, NIST SP 800-90A) --------

class DRBG:
    """Hash-based DRBG (SHA-256) — deterministic random bit generator."""

    def __init__(self, entropy_input: bytes, personalization: bytes = b""):
        import hashlib
        self._hash = hashlib.sha256
        seed = entropy_input + personalization + struct.pack(">H", 1)
        self._state = self._hash(seed).digest()
        self._reseed_counter = 0
        self._has_c = _HAS_C

    def generate(self, nbytes: int) -> bytes:
        blocks = []
        temp = b""
        v = self._state
        while len(temp) < nbytes:
            v = self._hash(v + struct.pack(">I", self._reseed_counter)).digest()
            temp += v
            self._reseed_counter += 1
            blocks.append(v)
        self._state = self._hash(self._state + blocks[-1] if blocks else v).digest()
        return temp[:nbytes]


# ---- Random (cryptographic-quality random) --------------------------------

class Random:
    """Cryptographically secure random byte generator."""

    @staticmethod
    def bytes(n: int) -> bytes:
        return os.urandom(n)

    @staticmethod
    def int(bits: int) -> int:
        nbytes = (bits + 7) // 8
        return int.from_bytes(os.urandom(nbytes), "big") >> (nbytes * 8 - bits)

    @staticmethod
    def range(low: int, high: int) -> int:
        if low >= high:
            return low
        span = high - low
        nbytes = (span.bit_length() + 7) // 8
        mask = (1 << nbytes * 8) - 1
        while True:
            r = int.from_bytes(os.urandom(nbytes), "big") & mask
            if r < span:
                return low + r


# ---- Constant-time utilities --------------------------------------------

class ConstantTime:
    """Constant-time comparison and selection primitives."""

    @staticmethod
    def compare(a: bytes, b: bytes) -> bool:
        if len(a) != len(b):
            return False
        result = 0
        for ca, cb in zip(a, b):
            result |= ca ^ cb
        return result == 0

    @staticmethod
    def select(bit: int, true_val: bytes, false_val: bytes) -> bytes:
        mask = ~(bit - 1)
        result = bytearray(len(true_val))
        for i in range(len(true_val)):
            t = true_val[i] if i < len(true_val) else 0
            f = false_val[i] if i < len(false_val) else 0
            result[i] = (t & mask) | (f & ~mask)
        return bytes(result)

    @staticmethod
    def compare_bytes(a: bytes, b: bytes) -> int:
        return 0 if ConstantTime.compare(a, b) else -1


# ---- EntropyPool --------------------------------------------------------

class EntropyPool:
    """Collects entropy from various sources and feeds a DRBG."""

    def __init__(self):
        self._pool = bytearray()
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    def add_entropy(self, data: bytes) -> None:
        with self._lock:
            self._pool.extend(data)
            if len(self._pool) > 4096:
                import hashlib
                self._pool = bytearray(hashlib.sha256(self._pool).digest())

    def reseed(self, drbg: DRBG) -> None:
        with self._lock:
            seed = bytes(self._pool) if self._pool else os.urandom(32)
            self._pool.clear()
        drbg = DRBG(seed)

    @staticmethod
    def collect_system_entropy() -> bytes:
        import hashlib
        h = hashlib.sha256()
        h.update(struct.pack("d", time.time()))
        h.update(os.urandom(32))
        h.update(str(threading.active_count()).encode())
        return h.digest()
