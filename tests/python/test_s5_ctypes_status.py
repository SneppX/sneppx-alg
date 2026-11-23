"""Regression tests: S5 ctypes bindings must honor C status and output lengths."""

import ctypes
from unittest.mock import Mock

import pytest

import SneppX_ALG.interface_bindings.s5_safety as s5


def _use_fake_lib(monkeypatch, lib):
    monkeypatch.setattr(s5, "_HAS_C_S5", True)
    monkeypatch.setattr(s5, "_LIB", lib)


def test_training_sanitize_fails_closed_on_error(monkeypatch):
    lib = Mock()
    lib.SNEPPX_training_sanitize = Mock(return_value=-1)
    _use_fake_lib(monkeypatch, lib)

    with pytest.raises(RuntimeError, match="training_sanitize"):
        s5.S5OutputVerifier().sanitize("hello")


def test_training_sanitize_honors_reported_length(monkeypatch):
    lib = Mock()

    def fake_sanitize(_text, _length, buf, out_len_ptr):
        out_len_ptr._obj.value = 4
        ctypes.memmove(buf, b"safe", 4)
        return 0

    lib.SNEPPX_training_sanitize = Mock(side_effect=fake_sanitize)
    _use_fake_lib(monkeypatch, lib)

    assert s5.S5OutputVerifier().sanitize("hello") == "safe"


def test_training_sanitize_rejects_oversized_length(monkeypatch):
    lib = Mock()

    def fake_sanitize(_text, _length, _buf, out_len_ptr):
        out_len_ptr._obj.value = 10**9
        return 0

    lib.SNEPPX_training_sanitize = Mock(side_effect=fake_sanitize)
    _use_fake_lib(monkeypatch, lib)

    with pytest.raises(RuntimeError, match="oversized"):
        s5.S5OutputVerifier().sanitize("hello")
