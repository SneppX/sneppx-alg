"""Per-key usage tracking with sliding-window counters and SQLite persistence."""

import time
import sqlite3
import os
import threading
import logging
from typing import Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

_USAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS request_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash    TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    tokens_in   INTEGER DEFAULT 0,
    tokens_out  INTEGER DEFAULT 0,
    endpoint    TEXT DEFAULT '',
    response_ms REAL DEFAULT 0,
    status_code INTEGER DEFAULT 200
);

CREATE INDEX IF NOT EXISTS idx_request_log_key_hash ON request_log(key_hash);
CREATE INDEX IF NOT EXISTS idx_request_log_ts ON request_log(timestamp);
"""


class UsageTracker:
    """Tracks request counts and token usage per API key.

    Maintains an in-memory sliding window for fast rate-limit checks
    and persists to SQLite for historical analytics.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or os.environ.get(
            "SNEPPX_USAGE_DB_PATH",
            os.path.join(os.path.expanduser("~"), ".sneppx", "usage.db"),
        )
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._local = threading.local()
        self._sliding: Dict[str, Tuple[float, int, int, int]] = {}
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript(_USAGE_SCHEMA)
        conn.commit()

    def check_rate_limit(self, key_hash: str, rpm: int) -> Tuple[bool, int]:
        """Check sliding-window rate limit. Returns (allowed, remaining)."""
        now = time.time()
        window = 60.0
        with self._lock:
            if key_hash in self._sliding:
                wstart, count, tokens_in, tokens_out = self._sliding[key_hash]
                if now - wstart > window:
                    self._sliding[key_hash] = (now, 1, 0, 0)
                    return True, rpm - 1
                if count >= rpm:
                    return False, 0
                self._sliding[key_hash] = (wstart, count + 1, tokens_in, tokens_out)
                return True, rpm - count - 1
            else:
                self._sliding[key_hash] = (now, 1, 0, 0)
                return True, rpm - 1

    def record(
        self,
        key_hash: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        endpoint: str = "",
        response_ms: float = 0,
        status_code: int = 200,
    ):
        """Record a request in the database."""
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO request_log (key_hash, timestamp, tokens_in, tokens_out, endpoint, response_ms, status_code) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key_hash, time.time(), tokens_in, tokens_out, endpoint, response_ms, status_code),
            )
            conn.commit()
        except Exception as e:
            logger.warning("Failed to record usage: %s", e)

    def get_stats(
        self, key_hash: str, since: Optional[float] = None
    ) -> Dict:
        """Get aggregate stats for a key."""
        conn = self._get_conn()
        if since:
            row = conn.execute(
                "SELECT COUNT(*) as requests, COALESCE(SUM(tokens_in),0) as tokens_in, "
                "COALESCE(SUM(tokens_out),0) as tokens_out, AVG(response_ms) as avg_ms "
                "FROM request_log WHERE key_hash = ? AND timestamp >= ?",
                (key_hash, since),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) as requests, COALESCE(SUM(tokens_in),0) as tokens_in, "
                "COALESCE(SUM(tokens_out),0) as tokens_out, AVG(response_ms) as avg_ms "
                "FROM request_log WHERE key_hash = ?",
                (key_hash,),
            ).fetchone()
        return {
            "requests": row[0],
            "tokens_in": row[1],
            "tokens_out": row[2],
            "avg_response_ms": round(row[3], 1) if row[3] else 0,
        }


_GLOBAL_TRACKER: Optional[UsageTracker] = None


def get_usage_tracker() -> UsageTracker:
    global _GLOBAL_TRACKER
    if _GLOBAL_TRACKER is None:
        _GLOBAL_TRACKER = UsageTracker()
    return _GLOBAL_TRACKER


def set_usage_tracker(tracker: UsageTracker):
    global _GLOBAL_TRACKER
    _GLOBAL_TRACKER = tracker


__all__ = ["UsageTracker", "get_usage_tracker", "set_usage_tracker"]
