"""Network-layer firewall controls with bounded state and IPv4/IPv6 support."""

from __future__ import annotations

import ipaddress
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Union

IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]
IPNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


@dataclass(frozen=True)
class CidrRule:
    network: IPNetwork
    allow: bool = True
    log: bool = True


@dataclass
class FirewallResult:
    allowed: bool = True
    status_code: int = 200
    reason: str = ""
    ring: int = 2


class SlidingWindowCounter:
    """Thread-safe sliding-window limiter with a hard cap on tracked keys."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
        max_keys: int = 100_000,
    ):
        if max_requests < 1:
            raise ValueError("max_requests must be at least 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than 0")
        if max_keys < 1:
            raise ValueError("max_keys must be at least 1")
        self.max_requests = int(max_requests)
        self.window_seconds = float(window_seconds)
        self.max_keys = int(max_keys)
        self._lock = threading.Lock()
        self._buckets: OrderedDict[str, Deque[float]] = OrderedDict()

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        if not isinstance(key, str) or not key:
            raise ValueError("rate-limit key must be a non-empty string")
        current = time.monotonic() if now is None else float(now)
        cutoff = current - self.window_seconds
        with self._lock:
            timestamps = self._buckets.get(key)
            if timestamps is None:
                if len(self._buckets) >= self.max_keys:
                    oldest_key, oldest_timestamps = next(iter(self._buckets.items()))
                    while oldest_timestamps and oldest_timestamps[0] <= cutoff:
                        oldest_timestamps.popleft()
                    if oldest_timestamps:
                        # Never evict an active client's history to admit a new key.
                        return False
                    self._buckets.pop(oldest_key)
                timestamps = deque()
                self._buckets[key] = timestamps
            else:
                self._buckets.move_to_end(key)

            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.max_requests:
                return False
            timestamps.append(current)
            return True

    @property
    def tracked_keys(self) -> int:
        with self._lock:
            return len(self._buckets)


class ConnectionTracker:
    """Bounded LRU-style source tracker; refreshes move entries to the newest end."""

    def __init__(self, max_connections: int = 1000, timeout: int = 300):
        if max_connections < 1:
            raise ValueError("max_connections must be at least 1")
        if timeout < 1:
            raise ValueError("timeout must be at least 1 second")
        self.max_connections = int(max_connections)
        self.timeout = int(timeout)
        self._lock = threading.Lock()
        self._table: OrderedDict[object, tuple[float, int]] = OrderedDict()

    def check(self, ip_int: object, now: Optional[float] = None) -> bool:
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            entry = self._table.get(ip_int)
            if entry is not None and current - entry[0] <= self.timeout:
                self._table[ip_int] = (current, entry[1] + 1)
                self._table.move_to_end(ip_int)
                return True
            if entry is not None:
                del self._table[ip_int]
            if len(self._table) >= self.max_connections:
                self._table.popitem(last=False)
            self._table[ip_int] = (current, 1)
            return True

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._table)


class FirewallNetwork:
    def __init__(
        self,
        allowlist: Optional[list[str]] = None,
        denylist: Optional[list[str]] = None,
        allowlist_enabled: Optional[bool] = None,
        denylist_enabled: Optional[bool] = None,
        max_connections: int = 1000,
        rate_limit_max: int = 100,
        rate_window_seconds: float = 60.0,
        rate_limit_enabled: bool = True,
        rate_limit: Optional[dict] = None,
        block_duration_seconds: int = 300,
        knock_ports: Optional[list[int]] = None,
        knock_window_ms: int = 5000,
        geoip_enabled: bool = False,
        geoip_db_path: Optional[str] = None,
        max_blocked_ips: int = 100_000,
        **kwargs,
    ):
        rate_config = rate_limit or {}
        rate_limit_max = rate_config.get("max_requests", rate_limit_max)
        rate_window_seconds = rate_config.get("window_seconds", rate_window_seconds)
        rate_limit_enabled = rate_config.get("enabled", rate_limit_enabled)

        self._allowlist = [self._parse_cidr(item) for item in (allowlist or [])]
        self._denylist = [self._parse_cidr(item) for item in (denylist or [])]
        self._allowlist_enabled = (
            bool(allowlist) if allowlist_enabled is None else bool(allowlist_enabled)
        )
        self._denylist_enabled = (
            bool(denylist) if denylist_enabled is None else bool(denylist_enabled)
        )
        self._rate_limit_enabled = bool(rate_limit_enabled)
        self._rate_limiter = SlidingWindowCounter(rate_limit_max, rate_window_seconds)
        self._conn_tracker = ConnectionTracker(max_connections, block_duration_seconds)
        if max_blocked_ips < 1:
            raise ValueError("max_blocked_ips must be at least 1")
        self._blocked_ips: OrderedDict[str, float] = OrderedDict()
        self._max_blocked_ips = int(max_blocked_ips)
        self._block_duration = int(block_duration_seconds)
        if self._block_duration < 1:
            raise ValueError("block_duration_seconds must be at least 1")
        self._state_lock = threading.Lock()
        self._knock_ports = list(knock_ports or [])
        if any(not isinstance(port, int) or not 1 <= port <= 65535 for port in self._knock_ports):
            raise ValueError("knock_ports must contain valid TCP/UDP port numbers")
        self._knock_window_ns = int(knock_window_ms) * 1_000_000
        if self._knock_window_ns < 1:
            raise ValueError("knock_window_ms must be greater than 0")
        self._knock_state: Dict[str, float] = {}
        self._geoip_enabled = bool(geoip_enabled)
        self._geoip_db_path = geoip_db_path

    @staticmethod
    def _parse_cidr(text: str) -> CidrRule:
        if not isinstance(text, str) or not text:
            raise ValueError("CIDR rule must be a non-empty string")
        allow = True
        value = text
        if value.startswith("+"):
            value = value[1:]
        elif value.startswith("-"):
            value = value[1:]
            allow = False
        try:
            network = ipaddress.ip_network(value, strict=False)
        except ValueError as exc:
            raise ValueError(f"invalid CIDR rule {text!r}: {exc}") from exc
        return CidrRule(network=network, allow=allow)

    @staticmethod
    def _parse_ip(ip_str: str) -> IPAddress:
        if not isinstance(ip_str, str) or not ip_str:
            raise ValueError("client IP must be a non-empty string")
        try:
            return ipaddress.ip_address(ip_str.strip())
        except ValueError as exc:
            raise ValueError(f"invalid client IP: {ip_str!r}") from exc

    @staticmethod
    def _ip_to_int(ip_str: str) -> int:
        """Backward-compatible IPv4 integer conversion."""
        address = FirewallNetwork._parse_ip(ip_str)
        if address.version != 4:
            raise ValueError("IPv4 address required")
        return int(address)

    def _check_allowlist(self, address: IPAddress) -> Optional[FirewallResult]:
        if not self._allowlist_enabled:
            return None
        if any(address.version == rule.network.version and address in rule.network for rule in self._allowlist):
            return None
        return FirewallResult(False, 403, "IP not in allowlist", 2)

    def _check_denylist(self, address: IPAddress) -> Optional[FirewallResult]:
        if not self._denylist_enabled:
            return None
        if any(address.version == rule.network.version and address in rule.network for rule in self._denylist):
            return FirewallResult(False, 403, "IP in denylist", 2)
        return None

    def _check_blocked(self, key: str, now: float) -> Optional[FirewallResult]:
        with self._state_lock:
            expiry = self._blocked_ips.get(key)
            if expiry is None:
                return None
            if now < expiry:
                self._blocked_ips.move_to_end(key)
                return FirewallResult(False, 429, "IP temporarily blocked", 2)
            del self._blocked_ips[key]
        return None

    def _check_rate_limit(self, ip_str: str, now: float) -> Optional[FirewallResult]:
        if not self._rate_limit_enabled or self._rate_limiter.allow(ip_str, now=now):
            return None
        with self._state_lock:
            if ip_str in self._blocked_ips:
                self._blocked_ips[ip_str] = now + self._block_duration
                self._blocked_ips.move_to_end(ip_str)
            elif len(self._blocked_ips) < self._max_blocked_ips:
                self._blocked_ips[ip_str] = now + self._block_duration
        return FirewallResult(False, 429, "rate limit exceeded", 2)

    def _check_port_knock(self, ip_str: str, knock_header: str) -> Optional[FirewallResult]:
        # This legacy header-based mechanism is compatibility-only. HTTP headers
        # are client-controlled and must never be treated as network-layer proof.
        if not self._knock_ports:
            return None
        return FirewallResult(
            False, 501,
            "port-knock enforcement requires trusted packet metadata; HTTP headers are not trusted",
            2,
        )

    def _check_geoip(self, ip_str: str) -> Optional[FirewallResult]:
        if not self._geoip_enabled:
            return None
        # No GeoIP provider is currently wired into this module. Never silently
        # report a pass when an operator has explicitly enabled this control.
        return FirewallResult(
            False, 503,
            "GeoIP enforcement is enabled but no verified GeoIP provider is configured",
            2,
        )

    def check_request(
        self,
        client_ip: str,
        now: Optional[float] = None,
        knock_header: str = "",
    ) -> FirewallResult:
        current = time.monotonic() if now is None else float(now)
        try:
            address = self._parse_ip(client_ip)
        except ValueError:
            return FirewallResult(False, 400, "invalid client IP", 2)
        key = str(address)

        result = self._check_blocked(key, current)
        if result is not None:
            return result
        # Deny rules always override allow rules, independent of input order.
        result = self._check_denylist(address)
        if result is not None:
            return result
        result = self._check_allowlist(address)
        if result is not None:
            return result
        if self._knock_ports:
            result = self._check_port_knock(key, knock_header)
            if result is not None:
                return result
        result = self._check_rate_limit(key, current)
        if result is not None:
            return result
        result = self._check_geoip(key)
        if result is not None:
            return result
        self._conn_tracker.check(key, current)
        return FirewallResult(True)

    def _check_connection_tracker(self, ip_int: object) -> Optional[FirewallResult]:
        """Backward-compatible hook for callers using the original tracker API."""
        self._conn_tracker.check(ip_int)
        return None

    @property
    def summary(self) -> dict:
        with self._state_lock:
            blocked_count = len(self._blocked_ips)
        return {
            "allowlist": [str(rule.network) for rule in self._allowlist],
            "denylist": [str(rule.network) for rule in self._denylist],
            "allowlist_enabled": self._allowlist_enabled,
            "denylist_enabled": self._denylist_enabled,
            "blocked_ips": blocked_count,
            "max_blocked_ips": self._max_blocked_ips,
            "rate_limit_enabled": self._rate_limit_enabled,
            "rate_limit_max": self._rate_limiter.max_requests,
            "rate_window_seconds": self._rate_limiter.window_seconds,
            "tracked_rate_limit_keys": self._rate_limiter.tracked_keys,
            "tracked_connections": self._conn_tracker.size,
            "knock_ports": list(self._knock_ports),
            "geoip_enabled": self._geoip_enabled,
            "has_asm": False,
        }
