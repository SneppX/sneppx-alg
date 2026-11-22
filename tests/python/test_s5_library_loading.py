"""Regression: S5 must not load native libraries from CWD or bare names."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import s5_safety


def test_no_env_means_no_native_library(monkeypatch):
    monkeypatch.delenv("SNEPPX_SECURITY_LIB", raising=False)
    assert s5_safety._find_library() is None


def test_relative_library_path_rejected(monkeypatch):
    monkeypatch.setenv("SNEPPX_SECURITY_LIB", "neural_security_c.dll")
    with pytest.raises(ValueError, match="absolute"):
        s5_safety._find_library()


def test_absolute_library_path_must_exist(monkeypatch, tmp_path):
    missing = tmp_path / "missing-lib.so"
    monkeypatch.setenv("SNEPPX_SECURITY_LIB", str(missing))
    with pytest.raises(FileNotFoundError):
        s5_safety._find_library()
