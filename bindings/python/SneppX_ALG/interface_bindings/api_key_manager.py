"""High-level API key manager — wraps ApiKeyStore with caching and business logic."""

import time
import hmac
import threading
from collections import OrderedDict
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

from .api_key_store import ApiKeyStore

logger = logging.getLogger(__name__)

_TIER_CONFIGS = {
    "free": {"rate_limit_rpm": 60, "max_keys": 2, "max_tokens_per_req": 2048},
    "pro": {"rate_limit_rpm": 1000, "max_keys": 10, "max_tokens_per_req": 8192},
    "enterprise": {"rate_limit_rpm": 10000, "max_keys": 100, "max_tokens_per_req": 65536},
}


@dataclass
class KeyCacheEntry:
    key_info: Dict[str, Any]
    expires_at: float


class ApiKeyManager:
    """High-level API key manager with in-memory cache and tier validation."""

    def __init__(
        self,
        store: Optional[ApiKeyStore] = None,
        cache_ttl: float = 60.0,
        max_cache_entries: int = 100_000,
    ):
        if cache_ttl <= 0 or max_cache_entries < 1:
            raise ValueError("cache_ttl and max_cache_entries must be positive")
        self._store = store or ApiKeyStore()
        self._cache_ttl = float(cache_ttl)
        self._max_cache_entries = int(max_cache_entries)
        self._cache: OrderedDict[str, KeyCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._admin_key: Optional[str] = None

    def set_admin_key(self, key: str):
        if not isinstance(key, str) or not key:
            raise ValueError("admin key must be a non-empty string")
        with self._lock:
            self._admin_key = key

    def is_admin_key(self, raw_key: str) -> bool:
        with self._lock:
            admin_key = self._admin_key
        return (
            admin_key is not None
            and isinstance(raw_key, str)
            and hmac.compare_digest(raw_key.encode("utf-8"), admin_key.encode("utf-8"))
        )

    def create_key(
        self,
        name: str = "",
        tier: str = "free",
        expires_in_days: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        tier = tier.lower()
        if tier not in _TIER_CONFIGS:
            logger.warning("Unknown tier '%s', defaulting to 'free'", tier)
            tier = "free"
        cfg = _TIER_CONFIGS[tier]
        result = self._store.create_key(
            name=name,
            tier=tier,
            rate_limit_rpm=cfg["rate_limit_rpm"],
            expires_in_days=expires_in_days,
            user_id=user_id,
        )
        return result

    def validate_key(self, raw_key: str) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_key, str) or not raw_key:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(raw_key)
            if entry is not None:
                if now < entry.expires_at:
                    self._cache.move_to_end(raw_key)
                    return dict(entry.key_info)
                del self._cache[raw_key]
            key_info = self._store.validate_key(raw_key)
            if not key_info:
                return None
            if len(self._cache) >= self._max_cache_entries:
                self._cache.popitem(last=False)
            cached = dict(key_info)
            self._cache[raw_key] = KeyCacheEntry(
                key_info=cached, expires_at=now + self._cache_ttl
            )
            return dict(cached)

    def revoke_key(self, key_id: str) -> bool:
        with self._lock:
            result = self._store.revoke_key(key_id)
            for raw_key, entry in list(self._cache.items()):
                if entry.key_info.get("id") == key_id:
                    del self._cache[raw_key]
            return result

    def list_keys(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        return self._store.list_keys(include_inactive=include_inactive)

    def get_key(self, key_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get_key(key_id)

    def update_key(
        self,
        key_id: str,
        name: Optional[str] = None,
        tier: Optional[str] = None,
        rate_limit_rpm: Optional[int] = None,
        expires_in_days: Optional[int] = None,
    ) -> bool:
        with self._lock:
            result = self._store.update_key(
                key_id, name=name, tier=tier,
                rate_limit_rpm=rate_limit_rpm, expires_in_days=expires_in_days,
            )
            for raw_key, entry in list(self._cache.items()):
                if entry.key_info.get("id") == key_id:
                    del self._cache[raw_key]
            return result

    def get_usage_stats(
        self, key_id: str, since: Optional[float] = None
    ) -> Dict[str, Any]:
        return self._store.get_usage_stats(key_id, since=since)

    def record_usage(
        self, raw_key: str, tokens_in: int = 0, tokens_out: int = 0, endpoint: str = ""
    ):
        self._store.record_usage(raw_key, tokens_in=tokens_in, tokens_out=tokens_out, endpoint=endpoint)

    def get_tier_config(self, tier: str) -> Dict[str, Any]:
        return dict(_TIER_CONFIGS.get(tier.lower(), _TIER_CONFIGS["free"]))


_GLOBAL_MANAGER: Optional[ApiKeyManager] = None


def get_key_manager() -> ApiKeyManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = ApiKeyManager()
    return _GLOBAL_MANAGER


def set_key_manager(manager: ApiKeyManager):
    global _GLOBAL_MANAGER
    _GLOBAL_MANAGER = manager


__all__ = [
    "ApiKeyManager",
    "get_key_manager",
    "set_key_manager",
]
