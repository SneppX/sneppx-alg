"""Symmetric-key cryptography bindings — AES-GCM, ChaCha20, Poly1305, AEAD.

Wraps the C implementations in ``security/crypto/c/`` with pure-Python
fallback via stdlib and ``cryptography``.
"""

import hmac
import os
from typing import Optional

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_security_c")


# ---- AES-GCM (NIST SP 800-38D) ------------------------------------------


class AESGCM:
    """AES-GCM authenticated encryption.

    Key sizes: 16 (AES-128), 24 (AES-192), 32 (AES-256).
    """

    def __init__(self, key: bytes):
        if len(key) not in (16, 24, 32):
            raise ValueError("AES-GCM key must be 16, 24, or 32 bytes")
        self._key = key
        self._has_c = _HAS_C

    @staticmethod
    def _iv(nonce: bytes) -> bytes:
        return nonce or os.urandom(12)

    def encrypt(self, plaintext: bytes, aad: bytes = b"", nonce: Optional[bytes] = None) -> bytes:
        iv = self._iv(nonce)
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _AESGCM

            cipher = _AESGCM(self._key)
            ct = cipher.encrypt(iv, plaintext, aad)
            return iv + ct
        except ImportError as exc:
            raise RuntimeError(
                "AES-GCM requires the 'cryptography' backend; refusing to emit unencrypted data"
            ) from exc

    def decrypt(self, ciphertext: bytes, aad: bytes = b"") -> Optional[bytes]:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _AESGCM

            cipher = _AESGCM(self._key)
            return cipher.decrypt(ciphertext[:12], ciphertext[12:], aad)
        except ImportError as exc:
            raise RuntimeError(
                "AEAD requires the 'cryptography' backend; refusing to skip tag verification"
            ) from exc


# ---- ChaCha20 (RFC 8439) -------------------------------------------------


class ChaCha20:
    """ChaCha20 stream cipher — 256-bit key, 96-bit nonce."""

    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("ChaCha20 key must be 32 bytes")
        self._key = key
        self._has_c = _HAS_C

    def encrypt(self, plaintext: bytes, aad: bytes = b"", nonce: Optional[bytes] = None) -> bytes:
        n = nonce or os.urandom(12)
        try:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

            cipher = ChaCha20Poly1305(self._key)
            ct = cipher.encrypt(n, plaintext, aad)
            return n + ct
        except ImportError as exc:
            raise RuntimeError(
                "ChaCha20-Poly1305 requires the 'cryptography' backend; refusing to emit unencrypted data"
            ) from exc

    def decrypt(self, ciphertext: bytes, aad: bytes = b"") -> Optional[bytes]:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

            cipher = ChaCha20Poly1305(self._key)
            return cipher.decrypt(ciphertext[:12], ciphertext[12:], aad)
        except ImportError as exc:
            raise RuntimeError(
                "AEAD requires the 'cryptography' backend; refusing to skip tag verification"
            ) from exc


# ---- Poly1305 (RFC 8439) -------------------------------------------------


class Poly1305:
    """Poly1305 one-time MAC."""

    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("Poly1305 key must be 32 bytes")
        self._key = key
        self._has_c = _HAS_C

    def mac(self, msg: bytes) -> bytes:
        """Compute the RFC 8439 Poly1305 tag for ``msg``."""
        if not isinstance(msg, (bytes, bytearray, memoryview)):
            raise TypeError("Poly1305 message must be bytes-like")
        data = bytes(msg)

        # RFC 8439 section 2.5.1: clamp r, treat s as the little-endian upper key half.
        r = int.from_bytes(self._key[:16], "little") & 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF
        s = int.from_bytes(self._key[16:], "little")
        modulus = (1 << 130) - 5

        acc = 0
        for offset in range(0, len(data), 16):
            block = data[offset : offset + 16]
            # Append the 0x01 stop bit in the byte following the final partial block.
            value = int.from_bytes(block, "little") | (1 << (8 * len(block)))
            acc = ((acc + value) * r) % modulus

        return ((acc + s) & ((1 << 128) - 1)).to_bytes(16, "little")

    def verify(self, msg: bytes, tag: bytes) -> bool:
        """Constant-time tag verification."""
        expected = self.mac(msg)
        if not isinstance(tag, (bytes, bytearray, memoryview)):
            return False
        return hmac.compare_digest(expected, bytes(tag))


# ---- Generic AEAD wrapper -----------------------------------------------


class AEAD:
    """Generic authenticated encryption with associated data.

    Supports AES-GCM and ChaCha20-Poly1305 via the ``algorithm`` parameter.
    """

    def __init__(self, key: bytes, algorithm: str = "aes-256-gcm"):
        self._algorithm = algorithm.lower()
        if self._algorithm.startswith("aes"):
            self._impl = AESGCM(key)
        elif self._algorithm.startswith("chacha20"):
            self._impl = ChaCha20(key)
        else:
            raise ValueError(f"Unknown AEAD algorithm: {algorithm}")

    def encrypt(self, plaintext: bytes, aad: bytes = b"", nonce: Optional[bytes] = None) -> bytes:
        return self._impl.encrypt(plaintext, aad, nonce)

    def decrypt(self, ciphertext: bytes, aad: bytes = b"") -> Optional[bytes]:
        return self._impl.decrypt(ciphertext, aad)
