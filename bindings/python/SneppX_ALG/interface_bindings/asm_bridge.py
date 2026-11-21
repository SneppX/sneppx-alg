"""ASM library bridge — loads MASM-compiled .dll, supports CPU feature detection.

Wraps assembly routines in ``security/crypto/asm/`` and ``security/firewall/asm/``.
"""

from typing import Optional, List, Dict
import platform
import struct
import ctypes
import threading

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_asm")


class CPUFeatures:
    """Detect CPU features (AES-NI, SHA-NI, AVX2, BMI2, ADX, SSE2)."""

    def __init__(self):
        self._features: Dict[str, bool] = {}
        self._detect()

    def _detect(self):
        self._features["aes_ni"] = self._check_bit(2, 25)   # ECX bit 25
        self._features["sha_ni"] = self._check_bit(7, 29)   # EBX bit 29 (leaf 7)
        self._features["avx2"] = self._check_bit(7, 5)      # EBX bit 5 (leaf 7)
        self._features["sse2"] = self._check_bit(1, 26)     # EDX bit 26
        self._features["bmi2"] = self._check_bit(7, 8)      # EBX bit 8 (leaf 7)
        self._features["adx"] = self._check_bit(7, 19)      # EBX bit 19 (leaf 7)

    @staticmethod
    def _check_bit(leaf: int, bit: int) -> bool:
        try:
            import cpuid
            if leaf == 1:
                _, _, ecx, edx = cpuid.cpuid(1)
                return bool(ecx & (1 << 25)) if bit >= 32 else bool(edx & (1 << bit))
            elif leaf == 7:
                _, ebx, _, _ = cpuid.cpuid(7, 0)
                return bool(ebx & (1 << bit))
            return False
        except (ImportError, AttributeError):
            return platform.machine() == "AMD64"

    def has(self, name: str) -> bool:
        return self._features.get(name, False)

    def supported(self) -> List[str]:
        return [k for k, v in self._features.items() if v]


class AsmFunction:
    """Wraps a function exported from the ASM shared library."""

    def __init__(self, lib: ctypes.CDLL, name: str, restype, *argtypes):
        self._name = name
        self._has_lib = lib is not None
        if self._has_lib and hasattr(lib, name):
            self._func = getattr(lib, name)
            self._func.restype = restype
            self._func.argtypes = argtypes
        else:
            self._func = None

    @property
    def available(self) -> bool:
        return self._func is not None

    def __call__(self, *args):
        if self._func is None:
            raise RuntimeError(f"ASM function '{self._name}' not available")
        return self._func(*args)


class AsmLibrary:
    """Loads and provides access to ASM-compiled shared library functions."""

    def __init__(self, lib_name: str = "neural_asm"):
        self._has_c = _HAS_C
        self._lib = _LIB
        self._lock = threading.Lock()
        self.cpu = CPUFeatures()

    @property
    def available(self) -> bool:
        return self._has_c

    def get_function(self, name: str, restype=ctypes.c_int,
                     *argtypes) -> AsmFunction:
        return AsmFunction(self._lib, name, restype, *argtypes)


class ConstantTime:
    """Constant-time operations using ASM when available, pure-Python fallback."""

    def __init__(self):
        self._asm = AsmLibrary()
        self._ct_cmp = None
        if self._asm.available:
            self._ct_cmp = self._asm.get_function(
                "sneppx_ct_compare_u64", ctypes.c_uint64,
                ctypes.c_uint64, ctypes.c_uint64)

    def compare_u64(self, a: int, b: int) -> bool:
        if self._ct_cmp and self._ct_cmp.available:
            return bool(self._ct_cmp(a, b))
        return a == b

    def compare_bytes(self, a: bytes, b: bytes) -> bool:
        if len(a) != len(b):
            return False
        result = 0
        for ca, cb in zip(a, b):
            result |= ca ^ cb
        return result == 0

    def select(self, cond: bool, true_val: int, false_val: int) -> int:
        mask = (1 << 64) - 1 if cond else 0
        return (true_val & mask) | (false_val & ~mask)


class SecureMemory:
    """Secure memory operations backed by ASM when available."""

    def __init__(self):
        self._asm = AsmLibrary()
        self._wipe_func = None
        if self._asm.available:
            self._wipe_func = self._asm.get_function(
                "sneppx_secure_wipe", None,
                ctypes.c_void_p, ctypes.c_size_t)

    def wipe(self, data: bytearray):
        n = len(data)
        if self._wipe_func and self._wipe_func.available:
            buf = (ctypes.c_uint8 * n).from_buffer(data)
            self._wipe_func(ctypes.cast(buf, ctypes.c_void_p), n)
        else:
            for i in range(n):
                data[i] = 0

    def wipe_buffer(self, buf):
        mv = memoryview(buf)
        if isinstance(buf, bytearray):
            self.wipe(buf)
        else:
            n = len(mv)
            for i in range(n):
                mv[i] = 0
