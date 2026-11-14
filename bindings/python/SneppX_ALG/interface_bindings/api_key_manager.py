"""High-level API key manager — wraps ApiKeyStore with caching and business logic."""

import time
import hmac
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

    def __init__(self, store: Optional[ApiKeyStore] = None, cache_ttl: float = 60.0):
        self._store = store or ApiKeyStore()
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, KeyCacheEntry] = {}
        self._admin_key: Optional[str] = None

    def set_admin_key(self, key: str):
        self._admin_key = key

    def is_admin_key(self, raw_key: str) -> bool:
        return (
            self._admin_key is not None
            and isinstance(raw_key, str)
            and hmac.compare_digest(raw_key.encode('utf-8'), self._admin_key.encode('utf-8'))
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
        if raw_key in self._cache:
            entry = self._cache[raw_key]
            if time.time() < entry.expires_at:
                return entry.key_info
            del self._cache[raw_key]
        key_info = self._store.validate_key(raw_key)
        if key_info:
            self._cache[raw_key] = KeyCacheEntry(
                key_info=key_info, expires_at=time.time() + self._cache_ttl
            )
        return key_info

    def revoke_key(self, key_id: str) -> bool:
        self._cache = {k: v for k, v in self._cache.items() if v.key_info.get("id") != key_id}
        return self._store.revoke_key(key_id)

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
        self._cache = {k: v for k, v in self._cache.items() if v.key_info.get("id") != key_id}
        return self._store.update_key(
            key_id, name=name, tier=tier,
            rate_limit_rpm=rate_limit_rpm, expires_in_days=expires_in_days,
        )

    def get_usage_stats(
        self, key_id: str, since: Optional[float] = None
    ) -> Dict[str, Any]:
        return self._store.get_usage_stats(key_id, since=since)

    def record_usage(
        self, raw_key: str, tokens_in: int = 0, tokens_out: int = 0, endpoint: str = ""
    ):
        self._store.record_usage(raw_key, tokens_in=tokens_in, tokens_out=tokens_out, endpoint=endpoint)

    def get_tier_config(self, tier: str) -> Dict[str, Any]:
        return _TIER_CONFIGS.get(tier.lower(), _TIER_CONFIGS["free"])


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
