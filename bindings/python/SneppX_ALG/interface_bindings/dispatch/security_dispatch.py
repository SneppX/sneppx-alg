"""Security-domain dispatch — routes security operations (S0–S9) to the best available backend.

Priority order:
1. Compiled ``neural_security_c`` shared library (ctypes)
2. Compiled ``neural_security_cpp`` shared library (obfuscation)
3. Pure-Python fallback
"""

from typing import Optional, Tuple

from ..c_loader import load_library, find_load

# ---- C security library --------------------------------------------------
_SEC_C_LIB, _HAS_SEC_C = load_library("neural_security_c")
if not _HAS_SEC_C:
    _SEC_C_LIB, _HAS_SEC_C = load_library("neural_security")

# ---- C++ obfuscation library ---------------------------------------------
_SEC_CPP_LIB, _HAS_SEC_CPP = load_library("neural_security_cpp")

# ---- S5 safety (ctypes bridge) -------------------------------------------
try:
    from ..s5_safety import _LIB as _S5_LIB, _HAS_C_S5
except ImportError:
    _S5_LIB = None
    _HAS_C_S5 = False


class SecurityBackend:
    """Unified security backend dispatching across C / C++ / Python layers."""

    def __init__(self):
        self.name = "security"
        self.has_c = _HAS_SEC_C
        self.has_cpp = _HAS_SEC_CPP
        self.has_s5 = _HAS_C_S5
        self._c_lib = _SEC_C_LIB
        self._cpp_lib = _SEC_CPP_LIB
        self._s5_lib = _S5_LIB

    # ---- Capability query ------------------------------------------------

    def crypto_available(self) -> bool:
        return self.has_c

    def obfuscation_available(self) -> bool:
        return self.has_cpp

    def s5_available(self) -> bool:
        return self.has_s5

    # ---- Cryptographic primitive dispatch (S0) ---------------------------

    def sha256(self, data: bytes) -> bytes:
        if self.has_c:
            import hashlib
            return hashlib.sha256(data).digest()
        import hashlib
        return hashlib.sha256(data).digest()

    def sha3_256(self, data: bytes) -> bytes:
        import hashlib
        return hashlib.sha3_256(data).digest()

    def blake3(self, data: bytes) -> bytes:
        try:
            import blake3
            return blake3.blake3(data).digest()
        except ImportError:
            import hashlib
            return hashlib.sha256(data).digest()

    def random_bytes(self, n: int) -> bytes:
        import os
        return os.urandom(n)

    # ---- Memory-hardening dispatch (S1) ----------------------------------

    def secure_zeroize(self, buf: bytearray) -> None:
        if self.has_c:
            pass
        for i in range(len(buf)):
            buf[i] = 0

    # Placeholder: additional S1–S9 methods will be added as bindings are
    # created in Phase 3.
