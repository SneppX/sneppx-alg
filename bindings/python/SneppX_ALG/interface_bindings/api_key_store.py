"""SQLite-backed API key storage — create, revoke, list, validate keys."""

import os
import sqlite3
import hashlib
import logging
import threading
import uuid
import time
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS api_keys (
    id          TEXT PRIMARY KEY,
    key_hash    TEXT NOT NULL UNIQUE,
    key_prefix  TEXT NOT NULL,
    name        TEXT NOT NULL DEFAULT '',
    tier        TEXT NOT NULL DEFAULT 'free',
    rate_limit_rpm INTEGER NOT NULL DEFAULT 60,
    is_active   INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    expires_at  REAL,
    last_used_at REAL,
    user_id     TEXT
);

CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);

CREATE TABLE IF NOT EXISTS key_usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash    TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    tokens_in   INTEGER DEFAULT 0,
    tokens_out  INTEGER DEFAULT 0,
    endpoint    TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_key_usage_key_hash ON key_usage(key_hash);
CREATE INDEX IF NOT EXISTS idx_key_usage_timestamp ON key_usage(timestamp);
"""


class ApiKeyStore:
    """Persistent SQLite-backed store for API keys and usage tracking."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or os.environ.get(
            "SNEPPX_KEY_STORE_PATH",
            os.path.join(os.path.expanduser("~"), ".sneppx", "api_keys.db"),
        )
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def create_key(
        self,
        name: str = "",
        tier: str = "free",
        rate_limit_rpm: int = 60,
        expires_in_days: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a new API key and store it. Returns the key details (including raw key)."""
        raw_key = f"sk-sneppx-{uuid.uuid4().hex}{uuid.uuid4().hex[:16]}"
        key_hash = self._hash_key(raw_key)
        key_prefix = raw_key[:18]
        now = time.time()
        expires_at = (now + expires_in_days * 86400) if expires_in_days else None

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO api_keys (id, key_hash, key_prefix, name, tier, rate_limit_rpm,
                                    is_active, created_at, expires_at, user_id)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
            (uuid.uuid4().hex, key_hash, key_prefix, name, tier,
             rate_limit_rpm, now, expires_at, user_id),
        )
        conn.commit()
        logger.info("Created API key %s (tier=%s, rpm=%d)", key_prefix, tier, rate_limit_rpm)
        return {
            "key": raw_key,
            "key_prefix": key_prefix,
            "name": name,
            "tier": tier,
            "rate_limit_rpm": rate_limit_rpm,
            "created_at": now,
            "expires_at": expires_at,
        }

    def validate_key(self, raw_key: str) -> Optional[Dict[str, Any]]:
        """Validate a raw API key. Returns key info dict if valid, None otherwise."""
        key_hash = self._hash_key(raw_key)
        conn = self._get_conn()
        row = conn.execute(
            """SELECT id, key_prefix, name, tier, rate_limit_rpm, is_active,
                     created_at, expires_at, last_used_at, user_id
               FROM api_keys WHERE key_hash = ?""",
            (key_hash,),
        ).fetchone()
        if row is None:
            return None
        key_info = dict(row)
        if not key_info["is_active"]:
            return None
        if key_info["expires_at"] and time.time() > key_info["expires_at"]:
            return None
        conn.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
            (time.time(), key_info["id"]),
        )
        conn.commit()
        return key_info

    def revoke_key(self, key_id: str) -> bool:
        """Revoke (deactivate) a key by ID. Returns True if found."""
        conn = self._get_conn()
        cur = conn.execute("UPDATE api_keys SET is_active = 0 WHERE id = ?", (key_id,))
        conn.commit()
        return cur.rowcount > 0

    def list_keys(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """List all keys (key_hash redacted)."""
        conn = self._get_conn()
        if include_inactive:
            rows = conn.execute(
                "SELECT id, key_prefix, name, tier, rate_limit_rpm, is_active, "
                "created_at, expires_at, last_used_at, user_id FROM api_keys ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, key_prefix, name, tier, rate_limit_rpm, is_active, "
                "created_at, expires_at, last_used_at, user_id FROM api_keys WHERE is_active=1 "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_key(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get key details by ID (key_hash redacted)."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, key_prefix, name, tier, rate_limit_rpm, is_active, "
            "created_at, expires_at, last_used_at, user_id FROM api_keys WHERE id = ?",
            (key_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_key(
        self,
        key_id: str,
        name: Optional[str] = None,
        tier: Optional[str] = None,
        rate_limit_rpm: Optional[int] = None,
        expires_in_days: Optional[int] = None,
    ) -> bool:
        """Update key fields. Returns True if found."""
        conn = self._get_conn()
        updates = []
        params = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if tier is not None:
            updates.append("tier = ?")
            params.append(tier)
        if rate_limit_rpm is not None:
            updates.append("rate_limit_rpm = ?")
            params.append(rate_limit_rpm)
        if expires_in_days is not None:
            updates.append("expires_at = ?")
            params.append(time.time() + expires_in_days * 86400)
        if not updates:
            return False
        params.append(key_id)
        cur = conn.execute(
            f"UPDATE api_keys SET {', '.join(updates)} WHERE id = ?", params
        )
        conn.commit()
        return cur.rowcount > 0

    def record_usage(
        self,
        raw_key: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        endpoint: str = "",
    ):
        """Record a usage event for a key."""
        key_hash = self._hash_key(raw_key)
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO key_usage (key_hash, timestamp, tokens_in, tokens_out, endpoint) "
            "VALUES (?, ?, ?, ?, ?)",
            (key_hash, time.time(), tokens_in, tokens_out, endpoint),
        )
        conn.commit()

    def get_usage_stats(
        self, key_id: str, since: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get usage statistics for a key."""
        conn = self._get_conn()
        key = self.get_key(key_id)
        if not key:
            return {}
        key_hash_field = "key_hash"
        query = f"SELECT COUNT(*) as requests, SUM(tokens_in) as tokens_in, SUM(tokens_out) as tokens_out FROM key_usage WHERE key_hash = (SELECT key_hash FROM api_keys WHERE id = ?)"
        params = [key_id]
        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since)
        row = conn.execute(query, params).fetchone()
        return {
            "key_id": key_id,
            "requests": row["requests"] or 0,
            "tokens_in": row["tokens_in"] or 0,
            "tokens_out": row["tokens_out"] or 0,
        }


__all__ = ["ApiKeyStore"]
