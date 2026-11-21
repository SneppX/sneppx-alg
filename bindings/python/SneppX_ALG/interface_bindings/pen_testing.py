"""Penetration testing bindings — S9: network fuzzer, self-audit.

Wraps C implementations in ``security/pentest/`` with pure-Python fallback.
"""

import os
import random
import struct
import threading
import time
from typing import Dict, List, Optional, Callable

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_security_c")


class NetworkFuzzer:
    """Protocol-aware network fuzzer — generates mutated inputs for robustness testing."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._mutations: List[str] = []
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    @staticmethod
    def _bit_flip(data: bytes, n_flips: int = 1) -> bytes:
        buf = bytearray(data)
        for _ in range(n_flips):
            if buf:
                byte_idx = random.randint(0, len(buf) - 1)
                bit_idx = random.randint(0, 7)
                buf[byte_idx] ^= (1 << bit_idx)
        return bytes(buf)

    @staticmethod
    def _random_bytes(n: int) -> bytes:
        return os.urandom(n)

    def fuzz(self, template: bytes, n: int = 100) -> List[bytes]:
        results = [self._bit_flip(template) for _ in range(n)]
        with self._lock:
            self._mutations.append(f"fuzz {n} variants from {len(template)}-byte template")
        return results

    def fuzz_protocol(self, template: bytes, protocol: str = "tcp") -> List[bytes]:
        variants = [template]
        variants.append(self._bit_flip(template, 2))
        variants.append(self._random_bytes(len(template)))
        variants.append(b"\x00" * len(template))
        variants.append(b"\xff" * len(template))
        variants.append(template + b"\x00" * 16)
        variants.append(template[:-1])
        with self._lock:
            self._mutations.append(f"fuzz_protocol {protocol}: {len(variants)} variants")
        return variants

    @property
    def mutation_log(self) -> List[str]:
        with self._lock:
            return list(self._mutations)


class SelfAuditor:
    """Self-audit and compliance checking.

    Runs predefined checks against the system to ensure security posture.
    """

    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._results: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    def register_check(self, name: str, check_fn: Callable[[], bool],
                       severity: str = "medium") -> None:
        with self._lock:
            self._checks[name] = (check_fn, severity)

    def run_check(self, name: str) -> Optional[dict]:
        with self._lock:
            entry = self._checks.get(name)
        if entry is None:
            return None
        check_fn, severity = entry
        try:
            passed = check_fn()
        except Exception as e:
            passed = False
        result = {
            "check": name,
            "passed": passed,
            "severity": severity,
            "timestamp": time.time(),
        }
        with self._lock:
            self._results[name] = result
        return result

    def run_all(self) -> Dict[str, dict]:
        with self._lock:
            names = list(self._checks.keys())
        for name in names:
            self.run_check(name)
        with self._lock:
            return dict(self._results)

    def summary(self) -> dict:
        with self._lock:
            total = len(self._results)
            passed = sum(1 for r in self._results.values() if r["passed"])
            return {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "compliance_pct": (passed / total * 100) if total > 0 else 100.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._results.clear()
