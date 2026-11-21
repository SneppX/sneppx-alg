"""Observability secret redaction.

Masks credentials, tokens, and private keys in log lines / structured records
before they are emitted, addressing the audit finding that observability output
lacked secret redaction.  Works on both plain strings and dict/JSON-able
structures.
"""

import re
import json

# High-signal value patterns (independent of key name).
_VALUE_RES = [
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),               # OpenAI-style keys
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                  # AWS access key id
    re.compile(r"\bayan-\w{20,}\b"),                       # generic token blobs
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),        # GitHub tokens
    re.compile(r"\b[a-f0-9]{32,}\b"),                      # long hex secrets
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),     # PEM private keys
]

# key = value / "key": "value" forms (case-insensitive).  Keys may be quoted
# (as in repr(dict)) and may use ':' or '='.  The auth/bearer forms may appear
# as "Authorization: Bearer <token>" or "Bearer <token>" with a space.
_KEY_VALUE_RES = [
    re.compile(
        r"((?:['\"]?)(?:api[_-]?key|apikey|secret|token|password|passwd|pwd|"
        r"private[_-]?key|access[_-]?key|client[_-]?secret|"
        r"connection[_-]?string|credential)\s*['\"]?\s*[:=]\s*)\S+",
        re.IGNORECASE,
    ),
    re.compile(
        r"((?:authorization|bearer)\s*[:=]?\s*(?:bearer\s+)?)\S+",
        re.IGNORECASE,
    ),
]

# Key names that always denote a secret, regardless of value shape.  Used to
# mask values even when the value itself has no recognisable high-signal
# pattern (e.g. a short password like "hunter2").
_SECRET_KEY_RES = re.compile(
    r"(api[_-]?key|apikey|secret|token|password|passwd|pwd|"
    r"authorization|bearer|private[_-]?key|access[_-]?key|client[_-]?secret|"
    r"connection[_-]?string|credential)",
    re.IGNORECASE,
)

_MASK = "***REDACTED***"


def _redact_string(text: str) -> str:
    out = text
    for rx in _VALUE_RES:
        out = rx.sub(_MASK, out)
    for rx in _KEY_VALUE_RES:
        out = rx.sub(lambda m: m.group(1) + _MASK, out)
    return out


def _is_secret_key(k) -> bool:
    return isinstance(k, str) and bool(_SECRET_KEY_RES.search(k))


def redact_secrets(obj, _mask: bool = False):
    """Return ``obj`` with secrets masked.  Accepts str, dict, list, or any
    object that can be JSON-serialised (after which the string is redacted).

    When a dict key is itself a secret key name, all of that value's strings are
    masked regardless of their shape (so e.g. ``{"password": "hunter2"}`` loses
    the value even though "hunter2" matches no high-signal pattern).
    """
    if isinstance(obj, str):
        return _MASK if _mask else _redact_string(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = redact_secrets(v, _mask=_is_secret_key(k))
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact_secrets(v, _mask=_mask) for v in obj)
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    # fall back to redacting the JSON representation
    try:
        return _redact_string(json.dumps(obj))
    except Exception:
        return _redact_string(str(obj))
