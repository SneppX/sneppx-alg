"""Model-artifact integrity — sign / verify / quarantine.

Provides non-forgeable (Ed25519) signing of serialized model artifacts and a
verify-or-quarantine helper.  This directly addresses the audit finding that
model-artifact integrity/quarantine was missing.  It uses the real Ed25519
implementation in :mod:`crypto_sign` (which falls back to the
``cryptography``-backed pure-Python path when the compiled C extension is
absent) — unlike the forgeable ``hash ^ 0xAA`` scheme in the C signed-update
path, Ed25519 signatures cannot be forged without the secret key.
"""

import os
import json

from .crypto_sign import Ed25519


class ArtifactIntegrityError(Exception):
    """Raised when an artifact fails integrity verification."""


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def sign_artifact(data: bytes, sk: bytes, key_id: str = "") -> dict:
    """Return a signature bundle ``{alg, key_id, signature(hex)}`` for ``data``."""
    sig = Ed25519.sign(sk, data)
    return {"alg": "ed25519", "key_id": key_id, "signature": sig.hex()}


def verify_artifact(data: bytes, pk: bytes, bundle: dict) -> bool:
    """Return ``True`` iff ``bundle`` is a valid Ed25519 signature of ``data``."""
    if not isinstance(bundle, dict) or bundle.get("alg") != "ed25519":
        return False
    try:
        sig = bytes.fromhex(bundle.get("signature", ""))
    except Exception:
        return False
    try:
        return bool(Ed25519.verify(pk, data, sig))
    except Exception:
        return False


def sign_file(path: str, sk: bytes, out_sig_path: str, key_id: str = "") -> dict:
    """Sign the bytes of ``path`` and write a ``.sig`` sidecar (JSON)."""
    bundle = sign_artifact(_read_bytes(path), sk, key_id=key_id)
    with open(out_sig_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f)
    return bundle


def verify_file(path: str, pk: bytes, sig_path: str = None) -> bool:
    """Verify ``path`` against its signature sidecar.

    If ``sig_path`` is ``None`` the sidecar is assumed to be ``path + ".sig"``.
    """
    sig_path = sig_path or (path + ".sig")
    if not os.path.exists(sig_path):
        return False
    with open(sig_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    return verify_artifact(_read_bytes(path), pk, bundle)


def verify_or_quarantine(path: str, pk: bytes, quarantine_dir: str,
                         sig_path: str = None) -> bool:
    """Verify ``path``; on failure move it (and its sidecar) into
    ``quarantine_dir`` and return ``False``.  Returns ``True`` if valid."""
    sig_path = sig_path or (path + ".sig")
    if verify_file(path, pk, sig_path):
        return True
    os.makedirs(quarantine_dir, exist_ok=True)
    base = os.path.basename(path)
    dest = os.path.join(quarantine_dir, base)
    # avoid clobbering an existing quarantined file
    i = 1
    while os.path.exists(dest):
        dest = os.path.join(quarantine_dir, f"{base}.{i}")
        i += 1
    os.replace(path, dest)
    if sig_path and os.path.exists(sig_path):
        os.replace(sig_path, os.path.join(quarantine_dir, os.path.basename(sig_path)))
    return False
