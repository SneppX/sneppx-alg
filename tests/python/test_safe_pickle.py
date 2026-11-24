"""Regression tests for the H-10 pickle opt-in gate."""

import io
import pickle

import pytest

from SneppX_ALG.interface_bindings.safe_pickle import (
    UnsafeDeserializationError,
    require_unsafe_opt_in,
    safe_pickle_load,
)


def _payload():
    return {"weights": [1, 2, 3], "step": 7}


def test_requires_explicit_opt_in():
    with pytest.raises(UnsafeDeserializationError, match="allow_unsafe=True"):
        require_unsafe_opt_in("test checkpoint", False)


def test_opt_in_is_a_runtime_warning_not_a_failure():
    with pytest.warns(RuntimeWarning, match="test checkpoint"):
        require_unsafe_opt_in("test checkpoint", True)


def test_safe_pickle_load_refuses_by_default():
    stream = io.BytesIO(pickle.dumps(_payload()))
    with pytest.raises(UnsafeDeserializationError):
        safe_pickle_load(stream, "test checkpoint")


def test_safe_pickle_load_returns_payload_when_opted_in():
    stream = io.BytesIO(pickle.dumps(_payload()))
    with pytest.warns(RuntimeWarning):
        result = safe_pickle_load(stream, "test checkpoint", allow_unsafe=True)
    assert result == _payload()


def test_error_is_a_runtime_error():
    assert issubclass(UnsafeDeserializationError, RuntimeError)
