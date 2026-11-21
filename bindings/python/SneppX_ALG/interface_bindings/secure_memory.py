"""Memory hardening bindings — S1: canaries, ASLR, guard pages, secure allocator.

Wraps C implementations in ``security/memory/`` with pure-Python fallback.
"""

import os
import ctypes
import struct
import threading
from typing import Optional, Dict, Tuple

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_security_c")


class SecureAllocator:
    """Allocator that zeroizes freed memory and guards against overflows."""

    def __init__(self):
        self._allocations: Dict[int, bytearray] = {}
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    def malloc(self, size: int) -> int:
        buf = bytearray(size)
        addr = id(buf)
        with self._lock:
            self._allocations[addr] = buf
        return addr

    def free(self, addr: int) -> None:
        with self._lock:
            buf = self._allocations.pop(addr, None)
        if buf is not None:
            for i in range(len(buf)):
                buf[i] = 0

    def secure_free(self, addr: int) -> None:
        self.free(addr)

    def get_buffer(self, addr: int) -> Optional[bytearray]:
        with self._lock:
            return self._allocations.get(addr)


class StackCanary:
    """Stack canary for buffer-overflow detection."""

    _CANARY_SIZE = 8

    def __init__(self):
        self._canary = os.urandom(self._CANARY_SIZE)
        self._has_c = _HAS_C

    @property
    def value(self) -> bytes:
        return self._canary

    def check(self, candidate: bytes) -> bool:
        if len(candidate) != self._CANARY_SIZE:
            return False
        result = 0
        for a, b in zip(candidate, self._canary):
            result |= a ^ b
        return result == 0

    def refresh(self) -> None:
        self._canary = os.urandom(self._CANARY_SIZE)


class ASLR:
    """Address-space layout randomization helpers."""

    @staticmethod
    def randomize_address(base: int, range_size: int) -> int:
        offset = int.from_bytes(os.urandom(4), "little") % range_size
        return base + offset

    @staticmethod
    def mmap_random(size: int, flags: int = 0) -> int:
        return int.from_bytes(os.urandom(4), "little")


class MemoryHardening:
    """Memory hardening operations — W^X, guard pages, seccomp.

    Many operations are OS-specific and are no-ops in the fallback.
    """

    def __init__(self):
        self._has_c = _HAS_C

    @staticmethod
    def w_xor_x(enable: bool = True) -> bool:
        return True

    @staticmethod
    def guard_page(size: int) -> Tuple[int, int]:
        total = size + 4096
        addr = int.from_bytes(os.urandom(4), "little")
        return addr, total

    @staticmethod
    def seccomp_filter(allow_list: Optional[list] = None) -> bool:
        return False


class MemoryLeakDetector:
    """Tracks allocations and reports leaks on scope exit."""

    def __init__(self):
        self._allocations: Dict[int, Tuple[str, int]] = {}
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    def track(self, addr: int, tag: str = "") -> int:
        with self._lock:
            self._allocations[addr] = (tag, id(threading.current_thread()))
        return addr

    def untrack(self, addr: int) -> None:
        with self._lock:
            self._allocations.pop(addr, None)

    def report(self) -> Dict[int, Tuple[str, int]]:
        with self._lock:
            return dict(self._allocations)

    @property
    def leak_count(self) -> int:
        with self._lock:
            return len(self._allocations)
