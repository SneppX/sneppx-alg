"""Explicit opt-in gate for pickle-based deserialization (audit finding H-10).

``pickle.load`` can execute arbitrary code while deserializing. Every
pickle-based loader in SneppX-Alg must therefore require the caller to pass
``allow_unsafe=True``. This module provides the shared gate used by the FSDP,
model, and trainer checkpoint loaders.
"""

from __future__ import annotations

import pickle
import warnings
from typing import Any, BinaryIO


class UnsafeDeserializationError(RuntimeError):
    """Raised when a pickle load is attempted without an explicit opt-in."""


def require_unsafe_opt_in(what: str, allow_unsafe: bool) -> None:
    """Raise unless the caller explicitly opted in to pickle deserialization."""
    if not allow_unsafe:
        raise UnsafeDeserializationError(
            f"Refusing to {what}: pickle-based checkpoints can execute "
            "arbitrary code. Only load trusted artifacts and pass "
            "allow_unsafe=True explicitly."
        )
    warnings.warn(
        f"Loading {what} with pickle is unsafe; only use trusted checkpoints.",
        RuntimeWarning,
        stacklevel=3,
    )


def safe_pickle_load(fileobj: BinaryIO, what: str, allow_unsafe: bool = False) -> Any:
    """Load a pickle only when ``allow_unsafe`` is explicitly True."""
    require_unsafe_opt_in(what, allow_unsafe)
    return pickle.load(fileobj)
