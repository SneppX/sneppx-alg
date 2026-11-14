"""User account store — registration, authentication, account management."""

import os
import sqlite3
import hashlib
import hmac
import secrets
import logging
import threading
import uuid
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_USER_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    tier            TEXT NOT NULL DEFAULT 'free',
    is_active       INTEGER NOT NULL DEFAULT 1,
    created_at      REAL NOT NULL,
    last_login_at   REAL
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE TABLE IF NOT EXISTS user_sessions (
    id              TEXT PRIMARY KEY,
    token_hash      TEXT NOT NULL UNIQUE,
    user_id         TEXT NOT NULL,
    created_at      REAL NOT NULL,
    expires_at      REAL NOT NULL,
    last_used_at    REAL,
    revoked_at      REAL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expiry ON user_sessions(expires_at);
"""


class UserStore:
    """SQLite-backed user account store with password authentication."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or os.environ.get(
            "SNEPPX_USER_DB_PATH",
            os.path.join(os.path.expanduser("~"), ".sneppx", "users.db"),
        )
        os.makedirs(os.path.dirname(os.path.abspath(self._db_path)), exist_ok=True)
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
        conn.executescript(_USER_SCHEMA)
        conn.commit()

    @staticmethod
    def _hash_password(password: str, salt: Optional[str] = None) -> str:
        if salt is None:
            salt = secrets.token_hex(16)
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return f"{salt}${h.hex()}"

    def _check_password(self, password: str, stored: str) -> bool:
        parts = stored.split("$")
        if len(parts) != 2:
            return False
        salt, _ = parts
        return hmac.compare_digest(self._hash_password(password, salt), stored)

    def create_user(self, email: str, password: str, tier: str = "free") -> Optional[Dict[str, Any]]:
        """Register a new user. Returns user info dict (without password_hash)."""
        if not isinstance(email, str) or not email.strip() or len(email.strip()) > 320:
            raise ValueError("email must be a non-empty string of at most 320 characters")
        if not isinstance(password, str) or len(password) < 8 or len(password) > 1024:
            raise ValueError("password must contain between 8 and 1024 characters")
        email = email.strip().casefold()
        conn = self._get_conn()
        user_id = uuid.uuid4().hex
        password_hash = self._hash_password(password)
        now = time.time()
        try:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, tier, is_active, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (user_id, email, password_hash, tier, now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            return None
        logger.info("Created user %s (%s)", user_id[:8], email)
        return {"id": user_id, "email": email, "tier": tier, "created_at": now}

    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user by email and password. Returns user info or None."""
        if not isinstance(email, str) or not isinstance(password, str):
            return None
        email = email.strip().casefold()
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, email, password_hash, tier, is_active, created_at, last_login_at "
            "FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        if row is None:
            return None
        user = dict(row)
        if not user["is_active"]:
            return None
        if not self._check_password(password, user["password_hash"]):
            return None
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (time.time(), user["id"]))
        conn.commit()
        del user["password_hash"]
        return user

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID (without password_hash)."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, email, tier, is_active, created_at, last_login_at "
            "FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, email, tier, is_active, created_at, last_login_at "
            "FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        return dict(row) if row else None

    def create_session_token(
        self, user_id: str, expires_in_seconds: int = 86400
    ) -> str:
        """Create a random, hashed, revocable login session token."""
        if not isinstance(expires_in_seconds, int) or not 60 <= expires_in_seconds <= 2_592_000:
            raise ValueError("session lifetime must be between 60 seconds and 30 days")
        conn = self._get_conn()
        user = conn.execute(
            "SELECT is_active FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if user is None or not user["is_active"]:
            raise ValueError("active user account required to create a session")
        token = "sxs_" + secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = time.time()
        conn.execute(
            "INSERT INTO user_sessions (id, token_hash, user_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, token_hash, user_id, now, now + expires_in_seconds),
        )
        conn.commit()
        return token

    def validate_session_token(
        self, token: str, now: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Validate a session token and return its active account metadata."""
        if not isinstance(token, str) or not token.startswith("sxs_"):
            return None
        current = time.time() if now is None else float(now)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        conn = self._get_conn()
        row = conn.execute(
            "SELECT s.id AS session_id, s.expires_at, u.id, u.email, u.tier, u.is_active "
            "FROM user_sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.revoked_at IS NULL",
            (token_hash,),
        ).fetchone()
        if row is None or not row["is_active"] or row["expires_at"] <= current:
            return None
        conn.execute(
            "UPDATE user_sessions SET last_used_at = ? WHERE id = ?",
            (current, row["session_id"]),
        )
        conn.commit()
        return {
            "id": row["id"],
            "user_id": row["id"],
            "email": row["email"],
            "tier": row["tier"],
            "session_id": row["session_id"],
            "expires_at": row["expires_at"],
            "token_type": "session",
        }

    def revoke_session_token(self, token: str) -> bool:
        """Revoke one session token without affecting the user's API keys."""
        if not isinstance(token, str) or not token.startswith("sxs_"):
            return False
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        conn = self._get_conn()
        cur = conn.execute(
            "UPDATE user_sessions SET revoked_at = ? "
            "WHERE token_hash = ? AND revoked_at IS NULL",
            (time.time(), token_hash),
        )
        conn.commit()
        return cur.rowcount > 0

    def revoke_user_sessions(self, user_id: str) -> int:
        """Revoke every active session for an account."""
        conn = self._get_conn()
        cur = conn.execute(
            "UPDATE user_sessions SET revoked_at = ? "
            "WHERE user_id = ? AND revoked_at IS NULL",
            (time.time(), user_id),
        )
        conn.commit()
        return cur.rowcount

    def update_tier(self, user_id: str, tier: str) -> bool:
        conn = self._get_conn()
        cur = conn.execute("UPDATE users SET tier = ? WHERE id = ?", (tier, user_id))
        conn.commit()
        return cur.rowcount > 0


_GLOBAL_USER_STORE: Optional[UserStore] = None


def get_user_store() -> UserStore:
    global _GLOBAL_USER_STORE
    if _GLOBAL_USER_STORE is None:
        _GLOBAL_USER_STORE = UserStore()
    return _GLOBAL_USER_STORE


def set_user_store(store: UserStore):
    global _GLOBAL_USER_STORE
    _GLOBAL_USER_STORE = store


__all__ = ["UserStore", "get_user_store", "set_user_store"]
