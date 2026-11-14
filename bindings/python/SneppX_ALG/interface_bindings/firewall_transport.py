import hashlib
import hmac
import os
import ssl
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FirewallResult:
    allowed: bool = True
    status_code: int = 200
    reason: str = ""
    ring: int = 1


class FirewallTransport:
    def __init__(
        self,
        tls_enabled: bool = False,
        certfile: Optional[str] = None,
        keyfile: Optional[str] = None,
        cafile: Optional[str] = None,
        require_client_cert: bool = False,
        min_version: int = ssl.TLSVersion.TLSv1_2,
        ciphers: Optional[str] = None,
        pinned_cert_fingerprints: Optional[list] = None,
        alpn_protocols: Optional[list] = None,
        **kwargs,
    ):
        self.tls_enabled = tls_enabled
        self.certfile = certfile
        self.keyfile = keyfile
        self.cafile = cafile
        self.require_client_cert = require_client_cert
        self.min_version = self._normalize_tls_version(min_version)
        self.ciphers = ciphers
        self.pinned_fingerprints = self._normalize_pins(pinned_cert_fingerprints or [])
        if self.require_client_cert and not self.tls_enabled:
            raise ValueError("client-certificate verification requires TLS to be enabled")
        if self.pinned_fingerprints and not self.tls_enabled:
            raise ValueError("certificate pinning requires TLS to be enabled")
        self.alpn_protocols = alpn_protocols or ["h2", "http/1.1"]
        self._ssl_context = None
        if self.tls_enabled and self.certfile and self.keyfile:
            self._build_context()

    @staticmethod
    def _normalize_tls_version(value):
        if isinstance(value, ssl.TLSVersion):
            version = value
        elif isinstance(value, int):
            legacy = {
                1: ssl.TLSVersion.TLSv1,
                2: ssl.TLSVersion.TLSv1_2,
                3: ssl.TLSVersion.TLSv1_3,
            }
            try:
                version = legacy[value] if value in legacy else ssl.TLSVersion(value)
            except ValueError as exc:
                raise ValueError(f"unsupported TLS minimum version: {value!r}") from exc
        elif isinstance(value, str):
            normalized = value.strip().lower().replace(".", "_").replace("-", "_")
            aliases = {
                "tlsv1_2": ssl.TLSVersion.TLSv1_2,
                "tls1_2": ssl.TLSVersion.TLSv1_2,
                "1_2": ssl.TLSVersion.TLSv1_2,
                "tlsv1_3": ssl.TLSVersion.TLSv1_3,
                "tls1_3": ssl.TLSVersion.TLSv1_3,
                "1_3": ssl.TLSVersion.TLSv1_3,
            }
            if normalized not in aliases:
                raise ValueError(f"unsupported TLS minimum version: {value!r}")
            version = aliases[normalized]
        else:
            raise ValueError(f"unsupported TLS minimum version: {value!r}")
        if version < ssl.TLSVersion.TLSv1_2:
            raise ValueError("TLS 1.2 or newer is required")
        return version

    @staticmethod
    def _normalize_pins(pins):
        normalized = []
        for fingerprint in pins:
            if not isinstance(fingerprint, str):
                raise ValueError("certificate fingerprints must be hexadecimal strings")
            value = fingerprint.replace(":", "").strip().lower()
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError("certificate fingerprints must be 64 hexadecimal characters")
            normalized.append(value)
        return normalized

    def _build_context(self) -> ssl.SSLContext:
        if not self.certfile or not os.path.isfile(self.certfile):
            raise FileNotFoundError(f"TLS certfile not found: {self.certfile}")
        if not self.keyfile or not os.path.isfile(self.keyfile):
            raise FileNotFoundError(f"TLS keyfile not found: {self.keyfile}")
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = self.min_version
        ctx.load_cert_chain(self.certfile, self.keyfile)
        if self.ciphers:
            ctx.set_ciphers(self.ciphers)
        if self.alpn_protocols:
            ctx.set_alpn_protocols(self.alpn_protocols)
        if self.require_client_cert:
            if not self.cafile or not os.path.isfile(self.cafile):
                raise FileNotFoundError(f"CA file required for mTLS: {self.cafile}")
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.load_verify_locations(self.cafile)
        self._ssl_context = ctx
        return ctx

    @property
    def ssl_context(self) -> Optional[ssl.SSLContext]:
        return self._ssl_context

    @staticmethod
    def _pinned_fingerprint(cert_der: bytes) -> str:
        return hashlib.sha256(cert_der).hexdigest()

    def verify_peer_certificate(self, cert_der: bytes) -> FirewallResult:
        if not self.require_client_cert and not self.pinned_fingerprints:
            return FirewallResult(allowed=True, ring=1)
        if not cert_der:
            return FirewallResult(
                allowed=False, status_code=403, reason="no client certificate", ring=1
            )
        fp = self._pinned_fingerprint(cert_der)
        if self.pinned_fingerprints:
            if not any(hmac.compare_digest(fp, pinned) for pinned in self.pinned_fingerprints):
                return FirewallResult(
                    allowed=False,
                    status_code=403,
                    reason="certificate fingerprint not pinned",
                    ring=1,
                )
        return FirewallResult(allowed=True, ring=1)

    def check_request(self, ssl_object: Optional[ssl.SSLObject] = None) -> FirewallResult:
        if not self.tls_enabled:
            return FirewallResult(allowed=True, ring=1)
        if ssl_object is None:
            return FirewallResult(allowed=False, status_code=426, reason="TLS required", ring=1)
        try:
            cert_der = ssl_object.getpeercert(binary_form=True)
        except Exception:
            cert_der = None
        return self.verify_peer_certificate(cert_der or b"")
