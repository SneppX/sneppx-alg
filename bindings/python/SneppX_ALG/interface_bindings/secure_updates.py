"""Secure update bindings — S7: signed updates, container security (SBOM, CVE).

Wraps C implementations in ``security/updates/`` with pure-Python fallback.

Signed updates use Ed25519 over a BLAKE3 digest of the update payload.
Verification is **fail-closed**: a trusted Ed25519 public key must be
registered via :meth:`set_trusted_public_key` and the signature must verify,
otherwise :meth:`apply` rejects the update. The previous HMAC-based path and
the constant-time hash-equality dev fallback are removed.
"""

import hashlib
import json
import threading
import time
from typing import Dict, List, Optional, Tuple

from .c_loader import load_library
from .crypto_sign import Ed25519
from .crypto_hash import Blake3

_LIB, _HAS_C = load_library("neural_security_c")


class SignedUpdate:
    """TUF-compliant signed update verification and application."""

    def __init__(self):
        self._updates: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._trusted_public_key: Optional[bytes] = None
        self._has_c = _HAS_C

    # -- hashing ----------------------------------------------------------
    @staticmethod
    def hash_package(data: bytes) -> bytes:
        """BLAKE3 digest of the update payload (32 bytes)."""
        return Blake3().hash(data)

    # -- trusted key management -------------------------------------------
    def set_trusted_public_key(self, public_key: bytes) -> None:
        """Register the trusted Ed25519 public key used to verify updates."""
        with self._lock:
            self._trusted_public_key = bytes(public_key)

    def get_trusted_public_key(self) -> Optional[bytes]:
        with self._lock:
            return self._trusted_public_key

    # -- signing / verification -------------------------------------------
    @staticmethod
    def sign_update(data: bytes, signing_key: bytes) -> Tuple[bytes, bytes]:
        """Sign ``data`` with Ed25519 over BLAKE3(data).

        Returns ``(data, signature)``.
        """
        digest = Blake3().hash(data)
        signature = Ed25519.sign(signing_key, digest)
        return data, signature

    @staticmethod
    def verify_signature(data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify an Ed25519 signature over BLAKE3(data) against ``public_key``."""
        digest = Blake3().hash(data)
        return Ed25519.verify(public_key, digest, signature)

    # -- update lifecycle -------------------------------------------------
    def register_update(
        self,
        update_id: str,
        metadata: dict,
        signature: Optional[bytes] = None,
    ) -> None:
        """Register an update pending verification and application.

        ``signature`` is the Ed25519 signature over BLAKE3 of the canonical
        metadata payload. Without a valid signature (and a registered trusted
        key) :meth:`apply` will reject the update.
        """
        with self._lock:
            self._updates[update_id] = {
                "metadata": metadata,
                "signature": signature,
                "registered_at": time.time(),
                "status": "pending",
            }

    def apply(self, update_id: str) -> bool:
        """Apply a registered update.

        Fail-closed: returns ``False`` unless the update is pending, a trusted
        public key is registered, a signature is on file, and the signature
        verifies against BLAKE3 of the canonical metadata payload.
        """
        with self._lock:
            entry = self._updates.get(update_id)
            if entry is None:
                return False
            if entry["status"] != "pending":
                return False
            if self._trusted_public_key is None:
                return False
            signature = entry.get("signature")
            if signature is None:
                return False
            payload = json.dumps(entry["metadata"], sort_keys=True).encode()
            digest = Blake3().hash(payload)
            if not Ed25519.verify(self._trusted_public_key, digest, signature):
                return False
            entry["status"] = "applied"
            entry["applied_at"] = time.time()
        return True

    def rollback(self, update_id: str) -> bool:
        with self._lock:
            entry = self._updates.get(update_id)
            if entry is None:
                return False
            entry["status"] = "rolled_back"
        return True

    def status(self, update_id: str) -> Optional[str]:
        with self._lock:
            entry = self._updates.get(update_id)
            return entry["status"] if entry else None


class ContainerSecurity:
    """Container security — SBOM generation, CVE scanning."""

    def __init__(self):
        self._sboms: Dict[str, dict] = {}
        self._vulns: Dict[str, List[dict]] = {}
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    @staticmethod
    def generate_sbom(packages: List[Dict[str, str]]) -> dict:
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "version": 1,
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "components": [],
        }
        for pkg in packages:
            sbom["components"].append({
                "type": "library",
                "name": pkg.get("name", "unknown"),
                "version": pkg.get("version", "0.0.0"),
                "purl": f"pkg:pypi/{pkg.get('name', 'unknown')}@{pkg.get('version', '0.0.0')}",
            })
        return sbom

    def register_sbom(self, image_id: str, sbom: dict) -> None:
        with self._lock:
            self._sboms[image_id] = sbom

    def scan_cves(self, image_id: str) -> List[dict]:
        with self._lock:
            return self._vulns.get(image_id, [])

    def add_vulnerability(self, image_id: str, vuln: dict) -> None:
        with self._lock:
            self._vulns.setdefault(image_id, []).append(vuln)

    def is_compliant(self, image_id: str, policy: dict) -> bool:
        vulns = self.scan_cves(image_id)
        max_critical = policy.get("max_critical", 0)
        critical_count = sum(1 for v in vulns if v.get("severity") == "critical")
        return critical_count <= max_critical
