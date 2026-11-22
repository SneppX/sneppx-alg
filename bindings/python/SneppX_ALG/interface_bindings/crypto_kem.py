"""Key-encapsulation mechanism (KEM) bindings — Kyber (ML-KEM), X25519.

Wraps the C implementations in ``security/crypto/c/`` with pure-Python
fallback via the ``cryptography`` package.
"""

import os
from typing import Optional, Tuple

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_security_c")


def _require_kyber_backend() -> None:
    """Refuse to fabricate ML-KEM keys/ciphertexts/shared secrets.

    The pure-Python bindings do not expose a validated ML-KEM implementation.
    Returning random bytes here used to make both peers derive unrelated
    secrets while reporting success, so callers must get an explicit error.
    """
    raise RuntimeError(
        "ML-KEM is not available through the Python bindings; refusing to "
        "fabricate keys, ciphertexts or shared secrets"
    )


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
        _require_kyber_backend()

    def encapsulate(self, pk: bytes) -> Tuple[bytes, bytes]:
        _require_kyber_backend()

    def decapsulate(self, sk: bytes, ct: bytes) -> Optional[bytes]:
        _require_kyber_backend()


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
        except ImportError as exc:
            raise RuntimeError(
                "X25519 requires the 'cryptography' backend; refusing to fabricate keys"
            ) from exc

    @staticmethod
    def dh(private: bytes, public: bytes) -> bytes:
        try:
            from cryptography.hazmat.primitives.asymmetric.x25519 import (
                X25519PrivateKey,
                X25519PublicKey,
            )

            sk = X25519PrivateKey.from_private_bytes(private)
            return sk.exchange(X25519PublicKey.from_public_bytes(public))
        except ImportError as exc:
            raise RuntimeError(
                "X25519 requires the 'cryptography' backend; refusing to fabricate shared secrets"
            ) from exc

    @staticmethod
    def scalar_mult(private: bytes, basepoint: bytes) -> bytes:
        return X25519.dh(private, basepoint)
