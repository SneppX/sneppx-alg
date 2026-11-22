"""Digital signature bindings — Ed25519, Dilithium (ML-DSA), SPHINCS+ (SLH-DSA).

The real implementations live in ``security/crypto/c/`` and are exposed to
Python through the pybind11 extension ``_SNEPPX_c`` (submodule ``crypto``).
When that extension is importable, every method below calls the real C
implementation.  A pure-Python fallback via the ``cryptography`` package is
kept only for environments where the compiled extension is unavailable.
"""

import os
import importlib.util
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Real C backend: the pybind11 extension built from bindings/python/.
# It is git-ignored (a compiled artifact) and lives next to this package.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_PYD = os.path.join(_PKG, "_SNEPPX_c.cp311-win_amd64.pyd")

_C = None
for _mod_name in (
    "SneppX_ALG._SNEPPX_c",
    "SneppX_ALG.interface_bindings._SNEPPX_c",
    "_SNEPPX_c",
):
    try:
        _mod = __import__(_mod_name, fromlist=["crypto"])
        _C = getattr(_mod, "crypto", None)
        if _C is not None:
            break
    except Exception:
        _C = None
if _C is None and os.path.exists(_PYD):
    try:
        _spec = importlib.util.spec_from_file_location("_SNEPPX_c", _PYD)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _C = getattr(_mod, "crypto", None)
    except Exception:
        _C = None

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_security_c")


# ---- Ed25519 (RFC 8032) -------------------------------------------------


class Ed25519:
    """Ed25519 signature scheme — sign / verify / keypair."""

    @staticmethod
    def keypair(seed: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        if _C is not None:
            # The C signing path expects the 64-byte (seed || pk) private
            # key, so always derive the public key from the seed and return
            # that form. Signing/verify are performed by the C backend.
            if seed is None:
                seed = os.urandom(32)
            from cryptography.hazmat.primitives.asymmetric import ed25519 as _ed

            pk = (
                _ed.Ed25519PrivateKey.from_private_bytes(seed)
                .public_key()
                .public_bytes_raw()
            )
            sk = bytes(seed[:32]) + pk
            return sk, pk
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519

            if seed:
                sk = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
            else:
                sk = ed25519.Ed25519PrivateKey.generate()
            pk = sk.public_key()
            return sk.private_bytes_raw(), pk.public_bytes_raw()
        except ImportError as exc:
            raise RuntimeError(
                "Ed25519 requires the native SneppX crypto extension or the "
                "'cryptography' package; refusing to generate placeholder keys"
            ) from exc

    @staticmethod
    def sign(sk: bytes, msg: bytes) -> bytes:
        if _C is not None and len(sk) >= 64:
            return _C.ed25519_sign(msg, bytes(sk[:64]))
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519

            priv = ed25519.Ed25519PrivateKey.from_private_bytes(sk)
            return priv.sign(msg)
        except ImportError as exc:
            raise RuntimeError(
                "Ed25519 signing requires the native SneppX crypto extension "
                "or the 'cryptography' package"
            ) from exc

    @staticmethod
    def verify(pk: bytes, msg: bytes, sig: bytes) -> bool:
        if _C is not None:
            return bool(_C.ed25519_verify(sig, msg, pk))
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519

            pub = ed25519.Ed25519PublicKey.from_public_bytes(pk)
            pub.verify(sig, msg)
            return True
        except Exception:
            return False


# ---- Dilithium (ML-DSA, FIPS 204) ---------------------------------------


class Dilithium:
    """CRYSTALS-Dilithium post-quantum signature scheme.

    Security levels: 2 (128-bit), 3 (192-bit), 5 (256-bit).  The C
    implementation uses pk = 32 + k*320 and sk = 96 + k*608, with
    k = 4 / 6 / 8 for modes 2 / 3 / 5 respectively.
    """

    MODE_ML_DSA_44 = 2
    MODE_ML_DSA_65 = 3
    MODE_ML_DSA_87 = 5

    def __init__(self, mode: int = MODE_ML_DSA_44):
        self.mode = mode

    def _params(self):
        k = {2: 4, 3: 6, 5: 8}.get(self.mode, 4)
        pk_size = 32 + k * 320
        sk_size = 96 + k * 608
        return pk_size, sk_size, 6000

    def keypair(self, seed: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        if _C is not None:
            pk, sk = _C.dilithium_keygen(self.mode)
            return pk, sk
        raise RuntimeError(
            "ML-DSA signing requires the native SneppX crypto extension; "
            "refusing to generate placeholder keys"
        )

    def sign(self, sk: bytes, msg: bytes) -> bytes:
        if _C is not None:
            return _C.dilithium_sign(sk, msg, self.mode)
        raise RuntimeError("ML-DSA signing requires the native SneppX crypto extension")

    def verify(self, pk: bytes, msg: bytes, sig: bytes) -> bool:
        if _C is not None:
            return _C.dilithium_verify(sig, msg, pk, self.mode) == 0
        return False


# ---- SPHINCS+ (SLH-DSA, FIPS 205) ---------------------------------------


class SphincsPlus:
    """SPHINCS+ stateless-hash post-quantum signature scheme.

    The C implementation uses a single fixed parameter set
    (pk = 32, sk = 64, sig = 14592); the ``variant`` argument is accepted
    for API compatibility but is currently ignored by the backend.
    """

    def __init__(self, variant: str = "128f"):
        self.variant = variant

    def _params(self):
        return 32, 64, 14592

    def keypair(self, seed: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        if _C is not None:
            pk, sk = _C.sphincs_keygen(0)
            # The C backend fills only the first 48 bytes
            # (sk_seed || sk_prf || pub_seed); the remaining 16 bytes of the
            # 64-byte secret key are unused padding. Normalise them to zero so
            # the key is deterministic instead of carrying uninitialised
            # memory from the native call.
            sk = bytes(sk[:48]) + b"\x00" * (64 - 48)
            return pk, sk
        raise RuntimeError(
            "SLH-DSA signing requires the native SneppX crypto extension; "
            "refusing to generate placeholder keys"
        )

    def sign(self, sk: bytes, msg: bytes) -> bytes:
        if _C is not None:
            return _C.sphincs_sign(sk, msg, 0)
        raise RuntimeError(
            "SLH-DSA signing requires the native SneppX crypto extension"
        )

    def verify(self, pk: bytes, msg: bytes, sig: bytes) -> bool:
        if _C is not None:
            return bool(_C.sphincs_verify(sig, msg, pk, 0))
        return False
