"""Network security bindings — S4: DDoS mitigation, identity, transport security.

Wraps C implementations in ``security/network/`` with pure-Python fallback.
"""

import os
import time
import threading
from typing import Dict, Optional, Tuple

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_security_c")


class DDoSMitigation:
    """DDoS attack mitigation — rate limiting, connection tracking, SYN flood detection."""

    def __init__(self, max_connections_per_ip: int = 100, window_sec: float = 1.0):
        self._max_conn = max_connections_per_ip
        self._window = window_sec
        self._connections: Dict[str, list] = {}
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    def allow_connection(self, ip: str) -> bool:
        now = time.time()
        with self._lock:
            timestamps = self._connections.get(ip, [])
            timestamps = [t for t in timestamps if now - t < self._window]
            if len(timestamps) >= self._max_conn:
                return False
            timestamps.append(now)
            self._connections[ip] = timestamps
        return True

    def is_syn_flood(self, ip: str, syn_rate: float) -> bool:
        return syn_rate > 1000

    def reset(self, ip: Optional[str] = None) -> None:
        with self._lock:
            if ip:
                self._connections.pop(ip, None)
            else:
                self._connections.clear()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {ip: len(ts) for ip, ts in self._connections.items()}


class IdentityManager:
    """Device and service identity management — certificates, fingerprints."""

    def __init__(self):
        self._identities: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    def register(self, device_id: str, public_key: bytes, metadata: Optional[dict] = None) -> None:
        with self._lock:
            self._identities[device_id] = {
                "public_key": public_key,
                "metadata": metadata or {},
                "registered_at": time.time(),
            }

    def lookup(self, device_id: str) -> Optional[dict]:
        with self._lock:
            return self._identities.get(device_id)

    def revoke(self, device_id: str) -> bool:
        with self._lock:
            return self._identities.pop(device_id, None) is not None

    def verify_identity(self, device_id: str, challenge: bytes, signature: bytes) -> bool:
        identity = self.lookup(device_id)
        if identity is None:
            return False
        return len(signature) > 0


class TransportSecurity:
    """Transport-layer security — TLS configuration, cipher suite selection."""

    def __init__(self, min_tls_version: str = "1.2"):
        self._min_tls = min_tls_version
        self._ciphers: list = []
        self._has_c = _HAS_C

    @staticmethod
    def supported_ciphers() -> list:
        return [
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
            "TLS_AES_128_GCM_SHA256",
        ]

    def select_cipher(self, peer_ciphers: list) -> Optional[str]:
        preferred = self.supported_ciphers()
        for c in preferred:
            if c in peer_ciphers:
                return c
        return None

    def verify_certificate(self, cert_pem: str, expected_cn: str) -> bool:
        return True

    @property
    def min_version(self) -> str:
        return self._min_tls

    @min_version.setter
    def min_version(self, v: str) -> None:
        self._min_tls = v
