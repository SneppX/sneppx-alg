"""Utility functions for the model hub."""

from __future__ import annotations

import hashlib
import os
from typing import Optional
import hmac
import base64


def compute_sha256(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


async def compute_sha256_async(file_path: str) -> str:
    """Compute SHA256 hash of a file asynchronously."""
    import asyncio

    def _hash():
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    return await asyncio.to_thread(_hash)


def compute_sha256_bytes(data: bytes) -> str:
    """Compute SHA256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def check_disk_space(path: str, required_bytes: int) -> bool:
    """Check if there's enough disk space at the given path."""
    import shutil
    usage = shutil.disk_usage(path)
    return usage.free >= required_bytes


def safe_join(base: str, *paths: str) -> str:
    """Safely join paths, preventing directory traversal."""
    final = os.path.normpath(os.path.join(base, *[p.lstrip("/") for p in paths]))
    final = os.path.abspath(final)
    base_abs = os.path.abspath(base)
    if not final.startswith(base_abs):
        raise ValueError(f"Path traversal detected: {final} is not under {base_abs}")
    return final


def format_file_size(size: int) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def get_cache_dir() -> str:
    """Get the sneppx cache directory."""
    base = os.environ.get("SNEPPX_HUB_CACHE")
    if base:
        return base

    home = os.path.expanduser("~")
    return os.path.join(home, ".cache", "sneppx", "hub")


def ensure_dir(path: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Hash a password using PBKDF2-SHA256. Returns 'salt:hash' hex string."""
    if salt is None:
        salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
    return salt.hex() + ":" + derived.hex()


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored hash."""
    try:
        salt_hex, hash_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
        return hmac.compare_digest(derived, expected)
    except (ValueError, AttributeError):
        return False
