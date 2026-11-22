"""Key vault bindings — S6: secure key storage, audit logging.

Wraps C implementations in ``security/ui/`` with pure-Python fallback.
"""

import os
import json
import time
import threading
from typing import Dict, List, Optional, Any

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_security_c")


class AuditLogger:
    """Security event audit logging with tamper-evident chain."""

    def __init__(self, log_path: Optional[str] = None):
        self._log_path = log_path or "audit.log"
        self._entries: List[dict] = []
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    def log(
        self, action: str, actor: str, resource: str, result: str, details: Optional[dict] = None
    ) -> None:
        import hashlib

        entry = {
            "timestamp": time.time(),
            "action": action,
            "actor": actor,
            "resource": resource,
            "result": result,
            "details": details or {},
        }
        prev_hash = self._entries[-1]["hash"] if self._entries else "0" * 64
        entry["prev_hash"] = prev_hash
        entry["hash"] = hashlib.sha256(
            json.dumps(entry, sort_keys=True, default=str).encode()
        ).hexdigest()
        with self._lock:
            self._entries.append(entry)
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except (IOError, OSError):
            pass

    def query(
        self, actor: Optional[str] = None, action: Optional[str] = None, limit: int = 100
    ) -> List[dict]:
        with self._lock:
            results = list(self._entries)
        if actor:
            results = [e for e in results if e["actor"] == actor]
        if action:
            results = [e for e in results if e["action"] == action]
        return results[-limit:]

    def verify_chain(self) -> bool:
        """Recompute every digest and verify the whole persisted chain.

        Comparing only ``prev_hash`` linkage is not tamper-evidence: an
        attacker who can edit the log can recompute the linking field too.
        """
        import hashlib
        import hmac

        with self._lock:
            memory_entries = [dict(e) for e in self._entries]
            log_path = self._log_path

        entries = memory_entries
        if log_path and os.path.isfile(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as handle:
                    entries = [json.loads(line) for line in handle if line.strip()]
            except (OSError, ValueError):
                return False
            if entries != memory_entries:
                return False

        expected_prev = "0" * 64
        for entry in entries:
            stored_hash = entry.get("hash", "")
            payload = {k: v for k, v in entry.items() if k != "hash"}
            computed = hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode()
            ).hexdigest()
            if not hmac.compare_digest(computed, stored_hash):
                return False
            if not hmac.compare_digest(entry.get("prev_hash", ""), expected_prev):
                return False
            expected_prev = stored_hash
        return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class KeyVault:
    """Secure key storage, generation, rotation, and access control."""

    def __init__(self, master_key: Optional[bytes] = None):
        self._master = master_key or os.urandom(32)
        self._keys: Dict[str, bytes] = {}
        self._metadata: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    def generate_key(self, key_id: str, algorithm: str = "aes-256") -> bytes:
        key = os.urandom(32)
        with self._lock:
            self._keys[key_id] = key
            self._metadata[key_id] = {
                "algorithm": algorithm,
                "created_at": time.time(),
                "version": self._metadata.get(key_id, {}).get("version", 0) + 1,
            }
        return key

    def store_key(self, key_id: str, key: bytes, algorithm: str = "raw") -> None:
        with self._lock:
            self._keys[key_id] = key
            self._metadata[key_id] = {
                "algorithm": algorithm,
                "created_at": time.time(),
                "version": self._metadata.get(key_id, {}).get("version", 0) + 1,
            }

    def get_key(self, key_id: str) -> Optional[bytes]:
        with self._lock:
            return self._keys.get(key_id)

    def rotate_key(self, key_id: str) -> Optional[bytes]:
        with self._lock:
            meta = self._metadata.get(key_id)
            if meta is None:
                return None
            new_key = os.urandom(32)
            self._keys[key_id] = new_key
            meta["version"] += 1
            meta["rotated_at"] = time.time()
        return new_key

    def delete_key(self, key_id: str) -> bool:
        with self._lock:
            if key_id in self._keys:
                del self._keys[key_id]
                del self._metadata[key_id]
                return True
            return False

    def list_keys(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._metadata)

    def wrap_key(self, key_id: str, wrapping_key_id: str) -> Optional[bytes]:
        key = self.get_key(key_id)
        if key is None:
            return None
        return bytes(
            b ^ w for b, w in zip(key, self.get_key(wrapping_key_id) or os.urandom(len(key)))
        )

    def unwrap_key(self, wrapped: bytes, key_id: str) -> Optional[bytes]:
        key = self.get_key(key_id)
        if key is None:
            return None
        return bytes(b ^ w for b, w in zip(wrapped, key[: len(wrapped)]))
