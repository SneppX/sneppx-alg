import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None

from .firewall_transport import FirewallTransport as _Transport
from .firewall_network import FirewallNetwork as _Network
from .firewall_application import FirewallApplication as _Application

_RESULT = "FirewallResult"


@dataclass
class FirewallConfig:
    enabled: bool = True
    block_all_by_default: bool = False
    transport: dict = field(default_factory=dict)
    network: dict = field(default_factory=dict)
    application: dict = field(default_factory=dict)


class FirewallRunner:
    def __init__(self, config: Optional[FirewallConfig] = None):
        cfg = config or FirewallConfig()
        self._cfg = cfg
        self._transport = _Transport(**cfg.transport) if cfg.transport else _Transport()
        self._network = _Network(**cfg.network) if cfg.network else _Network()
        self._application = _Application(**cfg.application) if cfg.application else _Application()
        self._allowed_count = 0
        self._denied_count = 0
        self._by_ring = {1: 0, 2: 0, 3: 0}
        self._stats_lock = threading.Lock()

    @property
    def config(self) -> FirewallConfig:
        return self._cfg

    @property
    def max_body_size(self) -> int:
        return self._application.max_body_size

    @property
    def max_header_count(self) -> int:
        return self._application.max_header_count

    def check_request(
        self,
        client_ip: str = "",
        method: str = "GET",
        path: str = "/",
        headers: dict = None,
        body: Optional[bytes] = None,
        query: str = "",
        ssl_object=None,
        knock_header: str = "",
    ) -> Any:
        if not self._cfg.enabled:
            return self._make_result(True)
        if headers is None:
            headers = {}

        result = self._transport.check_request(ssl_object)
        if not result.allowed:
            self._deny(result)
            return result

        result = self._network.check_request(client_ip, knock_header=knock_header)
        if not result.allowed:
            self._deny(result)
            return result

        if self._cfg.block_all_by_default and not self._application.is_path_allowed(path):
            result = self._make_result(
                False, 403, "blocked by default: path is not allowlisted", 3
            )
            self._deny(result)
            return result

        result = self._application.check_request(method, path, headers, body, query, client_ip)
        if not result.allowed:
            self._deny(result)
            return result

        with self._stats_lock:
            self._allowed_count += 1
        return self._make_result(True)

    def release_concurrent(self, client_ip: str):
        self._application.release_concurrent(client_ip)

    def _deny(self, result):
        ring = getattr(result, "ring", 2)
        with self._stats_lock:
            self._denied_count += 1
            self._by_ring[ring] = self._by_ring.get(ring, 0) + 1

    @staticmethod
    def _make_result(allowed: bool, status_code: int = 200, reason: str = "", ring: int = 2) -> Any:
        try:
            from .firewall_network import FirewallResult as FR
            return FR(allowed=allowed, status_code=status_code, reason=reason, ring=ring)
        except ImportError:
            class FR:
                def __init__(self, allowed, status_code, reason, ring):
                    self.allowed = allowed
                    self.status_code = status_code
                    self.reason = reason
                    self.ring = ring
            return FR(allowed, status_code, reason, ring)

    @property
    def summary(self) -> dict:
        with self._stats_lock:
            allowed = self._allowed_count
            denied = self._denied_count
            by_ring = {str(k): v for k, v in self._by_ring.items()}
        return {
            "enabled": self._cfg.enabled,
            "block_all_by_default": self._cfg.block_all_by_default,
            "allowed_paths": list(self._application.allowed_paths),
            "allowed": allowed,
            "denied": denied,
            "by_ring": by_ring,
            "transport": self._transport.tls_enabled,
            "network": self._network.summary,
        }


def _load_config(path: str = "") -> dict:
    if not path:
        candidates = [
            os.environ.get("FIREWALL_CONFIG", ""),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "security", "firewall", "firewall.yaml"),
        ]
    else:
        candidates = [path]
    for c in candidates:
        if c and os.path.isfile(c):
            if yaml is None:
                raise ImportError("PyYAML is required to load firewall config")
            with open(c, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    return {}


def create_firewall(
    config_path: str = "",
    env_overrides: bool = True,
    **cli_overrides,
) -> FirewallRunner:
    raw = _load_config(config_path)
    fw_cfg = raw.get("firewall", {}) if raw else {}
    cfg = FirewallConfig()

    cfg.enabled = fw_cfg.get("enabled", cfg.enabled)
    cfg.block_all_by_default = fw_cfg.get("block_all_by_default", cfg.block_all_by_default)
    cfg.transport = fw_cfg.get("transport", cfg.transport)
    cfg.network = fw_cfg.get("network", cfg.network)
    cfg.application = fw_cfg.get("application", cfg.application)

    if env_overrides:
        _apply_env_overrides(cfg)
    if cli_overrides:
        _apply_cli_overrides(cfg, cli_overrides)

    return FirewallRunner(cfg)


def _apply_env_overrides(cfg: FirewallConfig):
    env = os.environ
    if "FIREWALL_ENABLED" in env:
        cfg.enabled = env["FIREWALL_ENABLED"].lower() in ("1", "true", "yes")
    if "TLS_ENABLED" in env:
        cfg.transport["tls_enabled"] = env["TLS_ENABLED"].lower() in ("1", "true", "yes")
    if "TLS_CERTFILE" in env:
        cfg.transport["certfile"] = env["TLS_CERTFILE"]
    if "TLS_KEYFILE" in env:
        cfg.transport["keyfile"] = env["TLS_KEYFILE"]
    if "TLS_CAFILE" in env:
        cfg.transport["cafile"] = env["TLS_CAFILE"]
    if "FIREWALL_ALLOWLIST" in env:
        cfg.network["allowlist"] = [x.strip() for x in env["FIREWALL_ALLOWLIST"].split(",") if x.strip()]
    if "FIREWALL_DENYLIST" in env:
        cfg.network["denylist"] = [x.strip() for x in env["FIREWALL_DENYLIST"].split(",") if x.strip()]
    if "FIREWALL_RATE_LIMIT" in env:
        cfg.network.setdefault("rate_limit", {})["max_requests"] = int(env["FIREWALL_RATE_LIMIT"])
    if "FIREWALL_MAX_BLOCKED_IPS" in env:
        cfg.network["max_blocked_ips"] = int(env["FIREWALL_MAX_BLOCKED_IPS"])
    if "FIREWALL_BODY_SIZE" in env:
        cfg.application["max_body_size"] = int(env["FIREWALL_BODY_SIZE"])
    if "FIREWALL_ALLOWED_PATHS" in env:
        cfg.application["allowed_paths"] = [x.strip() for x in env["FIREWALL_ALLOWED_PATHS"].split(",") if x.strip()]


def _apply_cli_overrides(cfg: FirewallConfig, overrides: dict):
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "enabled":
            cfg.enabled = value
        elif key == "block_all_by_default":
            cfg.block_all_by_default = value
        elif key in ("tls_enabled", "certfile", "keyfile", "cafile", "require_client_cert", "min_version", "ciphers", "alpn_protocols"):
            cfg.transport[key] = value
        elif key in ("max_connections", "rate_limit_max", "rate_window_seconds", "block_duration_seconds", "max_blocked_ips", "knock_ports", "knock_window_ms", "geoip_enabled", "geoip_db_path", "allowlist", "denylist", "allowlist_enabled", "denylist_enabled"):
            cfg.network[key] = value
        elif key in ("max_body_size", "allowed_methods", "max_concurrent_per_ip", "enable_injection_filter", "allowed_paths"):
            cfg.application[key] = value
