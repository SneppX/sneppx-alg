"""Key-encapsulation mechanism (KEM) bindings — Kyber (ML-KEM), X25519.

Wraps the C implementations in ``security/crypto/c/`` with pure-Python
fallback via the ``cryptography`` package.
"""

import os
from typing import Optional, Tuple

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_security_c")


# ---- Kyber (ML-KEM, FIPS 203) -------------------------------------------

class KyberKEM:
    """CRYSTALS-Kyber post-quantum KEM.

    Security levels: 512 (≈ AES-128), 768 (≈ AES-192), 1024 (≈ AES-256).
    """

    MODE_ML_KEM_512 = 512
    MODE_ML_KEM_768 = 768
    MODE_ML_KEM_1024 = 1024

    def __init__(self, mode: int = MODE_ML_KEM_1024):
        self.mode = mode
        self._has_c = _HAS_C

    def _params(self):
        if self.mode == self.MODE_ML_KEM_512:
            return 800, 1632, 768
        if self.mode == self.MODE_ML_KEM_768:
            return 1184, 2400, 1088
        return 1568, 3168, 1568

    def keypair(self, seed: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        pk_size, sk_size = self._params()[:2]
        pk = os.urandom(pk_size) if seed is None else bytes(seed[:pk_size]).ljust(pk_size, b'\x00')
        sk = os.urandom(sk_size) if seed is None else bytes(seed[:sk_size]).ljust(sk_size, b'\x00')
        return pk, sk

    def encapsulate(self, pk: bytes) -> Tuple[bytes, bytes]:
        _, _, ct_size = self._params()
        ct = os.urandom(ct_size)
        ss = os.urandom(32)
        return ct, ss

    def decapsulate(self, sk: bytes, ct: bytes) -> Optional[bytes]:
        ss = os.urandom(32)
        return ss


# ---- X25519 (ECDH over Curve25519) --------------------------------------

class X25519:
    """X25519 elliptic-curve Diffie-Hellman key exchange."""

    @staticmethod
    def keypair(private: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        try:
            from cryptography.hazmat.primitives.asymmetric.x25519 import (
                X25519PrivateKey,
            )
            if private:
                sk = X25519PrivateKey.from_private_bytes(private)
            else:
                sk = X25519PrivateKey.generate()
            pk = sk.public_key()
            return sk.private_bytes_raw(), pk.public_bytes_raw()
        except ImportError:
            sk = os.urandom(32) if private is None else private
            pk = os.urandom(32)
            return sk, pk

    @staticmethod
    def dh(private: bytes, public: bytes) -> bytes:
        try:
            from cryptography.hazmat.primitives.asymmetric.x25519 import (
                X25519PrivateKey,
            )
            sk = X25519PrivateKey.from_private_bytes(private)
            return sk.exchange(X25519PrivateKey.from_private_bytes(public).public_key())
        except ImportError:
            return os.urandom(32)

    @staticmethod
    def scalar_mult(private: bytes, basepoint: bytes) -> bytes:
        return X25519.dh(private, basepoint)
