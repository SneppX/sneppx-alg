"""Application-layer request validation for the SneppX firewall."""

from __future__ import annotations

import posixpath
import re
import urllib.parse
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional, Set


_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+.^_`|~0-9a-z-]+$")


@dataclass
class FirewallResult:
    allowed: bool = True
    status_code: int = 200
    reason: str = ""
    ring: int = 3


_INJECTION_PATTERNS = (
    re.compile(r"(?is)\bunion\b.{0,200}\bselect\b|\bselect\b.{0,200}\bfrom\b"),
    re.compile(r"(?i)\b(?:drop\s+(?:table|database)|delete\s+from)\b"),
    re.compile(r"(?i)\b(?:exec|execute)\s*(?:\(|\b)|\bxp_cmdshell\b|\bsp_executesql\b"),
    re.compile(r"(?i)\b(?:or|and)\s+['\"\w]+\s*=\s*['\"\w]+"),
    re.compile(r"(?is)<\s*script\b|</\s*script\s*>|javascript\s*:"),
    re.compile(r"(?i)(?:;|&&|\|\|)\s*(?:/bin/(?:sh|bash)|powershell(?:\.exe)?|cmd(?:\.exe)?\s+/c)\b"),
)


class ConcurrentLimiter:
    """Per-client concurrency accounting without semaphore over-release or key leaks."""

    def __init__(self, max_per_ip: int = 10, max_clients: int = 100_000):
        if max_per_ip < 1:
            raise ValueError("max_per_ip must be at least 1")
        if max_clients < 1:
            raise ValueError("max_clients must be at least 1")
        self.max_per_ip = int(max_per_ip)
        self.max_clients = int(max_clients)
        self._lock = Lock()
        self._counts: Dict[str, int] = {}

    def acquire(self, ip: str) -> bool:
        if not ip:
            return False
        with self._lock:
            count = self._counts.get(ip, 0)
            if count >= self.max_per_ip:
                return False
            if ip not in self._counts and len(self._counts) >= self.max_clients:
                return False
            self._counts[ip] = count + 1
            return True

    def release(self, ip: str) -> None:
        with self._lock:
            count = self._counts.get(ip, 0)
            if count <= 1:
                self._counts.pop(ip, None)
            else:
                self._counts[ip] = count - 1

    @property
    def tracked_clients(self) -> int:
        with self._lock:
            return len(self._counts)


class FirewallApplication:
    def __init__(
        self,
        max_body_size: int = 1_048_576,
        allowed_methods: Optional[Set[str]] = None,
        allowed_content_types: Optional[Set[str]] = None,
        max_concurrent_per_ip: int = 10,
        enable_injection_filter: bool = True,
        max_path_length: int = 8192,
        max_query_length: int = 16384,
        max_header_count: int = 100,
        max_header_value_length: int = 8192,
        allowed_paths: Optional[list[str]] = None,
        **kwargs,
    ):
        if max_body_size < 0:
            raise ValueError("max_body_size cannot be negative")
        if min(max_path_length, max_query_length, max_header_count, max_header_value_length) < 1:
            raise ValueError("request limits must be positive")
        self.max_body_size = int(max_body_size)
        self.allowed_methods = {m.upper() for m in (allowed_methods or {"GET", "POST"})}
        self.allowed_content_types = {
            item.lower() for item in (allowed_content_types or {"application/json"})
        }
        self.max_concurrent_per_ip = int(max_concurrent_per_ip)
        self.enable_injection_filter = bool(enable_injection_filter)
        self.max_path_length = int(max_path_length)
        self.max_query_length = int(max_query_length)
        self.max_header_count = int(max_header_count)
        self.max_header_value_length = int(max_header_value_length)
        if allowed_paths is not None and not isinstance(allowed_paths, (list, tuple)):
            raise ValueError("allowed_paths must be a list or tuple of path patterns")
        self.allowed_paths = list(allowed_paths or [])
        self._allowed_exact = set()
        self._allowed_prefixes = []
        for pattern in self.allowed_paths:
            if not isinstance(pattern, str) or not pattern.startswith("/"):
                raise ValueError("allowed path patterns must be absolute paths")
            wildcard = pattern.endswith("/*")
            base = pattern[:-2] if wildcard else pattern
            if wildcard and not base:
                base = "/"
            if "*" in base or (not wildcard and "*" in pattern):
                raise ValueError("only a trailing /* path wildcard is supported")
            normalized = self._normalize_path(base)
            if normalized is None:
                raise ValueError(f"invalid allowed path pattern: {pattern!r}")
            if wildcard:
                self._allowed_prefixes.append(normalized.rstrip("/") or "/")
            else:
                self._allowed_exact.add(normalized)
        self._concurrent = ConcurrentLimiter(self.max_concurrent_per_ip)
        self._suspicious_headers = {
            "x-forwarded-for",
            "x-forwarded-host",
            "x-http-method-override",
            "x-original-url",
            "x-rewrite-url",
        }

    @staticmethod
    def _normalize_headers(headers: dict) -> Optional[dict]:
        normalized = {}
        for key, value in headers.items():
            if not isinstance(key, str):
                return None
            name = key.strip().lower()
            if not _HEADER_NAME_PATTERN.fullmatch(name):
                return None
            if name in normalized:
                return None
            if isinstance(value, bytes):
                try:
                    value = value.decode("latin-1")
                except UnicodeDecodeError:
                    return None
            if not isinstance(value, str):
                value = str(value)
            if "\r" in value or "\n" in value or "\x00" in value:
                return None
            normalized[name] = value.strip()
        return normalized

    def _check_body_size(self, body_length: int) -> Optional[FirewallResult]:
        if body_length > self.max_body_size:
            return FirewallResult(False, 413, "request body too large", 3)
        return None

    def _check_method(self, method: str) -> Optional[FirewallResult]:
        if not isinstance(method, str) or not method or not method.isascii():
            return FirewallResult(False, 400, "invalid request method", 3)
        normalized = method.upper()
        if normalized not in self.allowed_methods:
            return FirewallResult(False, 405, f"method {normalized} not allowed", 3)
        return None

    def _check_content_type(self, content_type: str) -> Optional[FirewallResult]:
        if not content_type:
            return None
        base_type = content_type.split(";", 1)[0].strip().lower()
        if base_type not in self.allowed_content_types:
            return FirewallResult(False, 415, f"content type {base_type} not allowed", 3)
        return None

    def _check_headers(self, headers: dict) -> Optional[FirewallResult]:
        if len(headers) > self.max_header_count:
            return FirewallResult(False, 431, "too many request headers", 3)
        for name, value in headers.items():
            if len(value) > self.max_header_value_length:
                return FirewallResult(False, 431, f"header {name} is too large", 3)
            if name in self._suspicious_headers:
                return FirewallResult(False, 400, f"untrusted forwarding header {name}", 3)
            if any(ord(ch) < 32 and ch != "\t" for ch in value):
                return FirewallResult(False, 400, "control character in request header", 3)
        host = headers.get("host", "")
        if host and (any(ch.isspace() for ch in host) or any(ch in host for ch in "#<>\\/@")):
            return FirewallResult(False, 400, "malformed host header", 3)
        content_length = headers.get("content-length")
        if content_length is not None:
            if len(content_length) > 20 or not content_length.isdecimal():
                return FirewallResult(False, 400, "invalid content-length header", 3)
            if int(content_length) > self.max_body_size:
                return FirewallResult(False, 413, "request body too large", 3)
        return None

    def _normalize_path(self, path: str) -> Optional[str]:
        if not isinstance(path, str) or not path or len(path) > self.max_path_length:
            return None
        raw_path = path.split("?", 1)[0]
        if not raw_path.startswith("/"):
            return None
        if re.search(r"%(?![0-9a-fA-F]{2})", raw_path):
            return None
        decoded = raw_path
        try:
            for _ in range(4):
                expanded = urllib.parse.unquote(decoded, errors="strict")
                if expanded == decoded:
                    break
                decoded = expanded
        except (UnicodeDecodeError, ValueError):
            return None
        if "%" in decoded and re.search(r"%(?:2e|2f|5c|00)", decoded, re.IGNORECASE):
            return None
        if "\x00" in decoded or "\\" in decoded or any(ord(ch) < 32 for ch in decoded):
            return None
        if any(segment == ".." for segment in decoded.split("/")):
            return None
        normalized = posixpath.normpath(decoded)
        if not normalized.startswith("/"):
            return None
        return normalized

    def is_path_allowed(self, path: str) -> bool:
        """Match a canonical path against exact routes and trailing /* prefixes."""
        normalized = self._normalize_path(path)
        if normalized is None:
            return False
        if normalized in self._allowed_exact:
            return True
        for prefix in self._allowed_prefixes:
            if prefix == "/" or normalized == prefix or normalized.startswith(prefix + "/"):
                return True
        return False

    def _check_path(self, path: str) -> Optional[FirewallResult]:
        normalized = self._normalize_path(path)
        if normalized is None:
            return FirewallResult(False, 400, "invalid or unsafe request path", 3)
        return None

    def _check_injection(self, body: Optional[bytes], query: str) -> Optional[FirewallResult]:
        if not self.enable_injection_filter:
            return None
        targets = []
        if body:
            if not isinstance(body, bytes):
                return FirewallResult(False, 400, "request body must be bytes", 3)
            targets.append(body.decode("utf-8", errors="replace"))
        if query:
            if not isinstance(query, str) or len(query) > self.max_query_length:
                return FirewallResult(False, 414, "request query is too large", 3)
            decoded_query = query
            for _ in range(3):
                expanded = urllib.parse.unquote(decoded_query)
                if expanded == decoded_query:
                    break
                decoded_query = expanded
            targets.append(decoded_query)
        for target in targets:
            for pattern in _INJECTION_PATTERNS:
                if pattern.search(target):
                    return FirewallResult(False, 400, "suspicious injection pattern detected", 3)
        return None

    def check_request(
        self,
        method: str,
        path: str,
        headers: Optional[dict],
        body: Optional[bytes] = None,
        query: str = "",
        client_ip: str = "",
    ) -> FirewallResult:
        result = self._check_method(method)
        if result is not None:
            return result
        normalized_headers = self._normalize_headers(headers or {})
        if normalized_headers is None:
            return FirewallResult(False, 400, "malformed or duplicate request header", 3)
        result = self._check_path(path)
        if result is not None:
            return result
        result = self._check_headers(normalized_headers)
        if result is not None:
            return result
        result = self._check_content_type(normalized_headers.get("content-type", ""))
        if result is not None:
            return result
        if body is not None:
            if not isinstance(body, bytes):
                return FirewallResult(False, 400, "request body must be bytes", 3)
            result = self._check_body_size(len(body))
            if result is not None:
                return result
            declared_length = normalized_headers.get("content-length")
            if declared_length is not None and int(declared_length) != len(body):
                return FirewallResult(False, 400, "content-length does not match request body", 3)
        result = self._check_injection(body, query)
        if result is not None:
            return result
        if client_ip and not self._concurrent.acquire(client_ip):
            return FirewallResult(False, 429, "too many concurrent requests", 3)
        return FirewallResult(True)

    def release_concurrent(self, ip: str) -> None:
        self._concurrent.release(ip)
