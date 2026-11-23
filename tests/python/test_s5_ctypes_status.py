"""Regression tests: S5 ctypes bindings must honor C status and output lengths."""

import ctypes
from unittest.mock import Mock

import numpy as np
import pytest

import SneppX_ALG.interface_bindings.s5_safety as s5


def _use_fake_lib(monkeypatch, lib):
    monkeypatch.setattr(s5, "_HAS_C_S5", True)
    monkeypatch.setattr(s5, "_LIB", lib)


def test_model_inversion_uses_in_place_c_output(monkeypatch):
    lib = Mock()

    def fake_defend(gradients, count, _noise, _clip):
        assert count == 2
        ctypes.memmove(gradients, (ctypes.c_double * 2)(7.0, 8.0), 16)
        return 0

    lib.SNEPPX_model_inversion_defend_gradients = Mock(side_effect=fake_defend)
    _use_fake_lib(monkeypatch, lib)

    out = s5.S5RLHFSafety().model_inversion_defense(np.array([1.0, 2.0]))
    assert np.allclose(out, [7.0, 8.0])
    lib.SNEPPX_model_inversion_defend_gradients.assert_called_once()
    assert len(lib.SNEPPX_model_inversion_defend_gradients.call_args.args) == 4


def test_model_inversion_fails_closed_on_error(monkeypatch):
    lib = Mock()
    lib.SNEPPX_model_inversion_defend_gradients = Mock(return_value=-1)
    _use_fake_lib(monkeypatch, lib)

    with pytest.raises(RuntimeError, match="model_inversion"):
        s5.S5RLHFSafety().model_inversion_defense(np.array([1.0, 2.0]))


def test_membership_defense_fails_closed_on_error(monkeypatch):
    lib = Mock()
    lib.SNEPPX_membership_inference_defense_apply = Mock(return_value=-1)
    _use_fake_lib(monkeypatch, lib)

    with pytest.raises(RuntimeError, match="membership_inference"):
        s5.S5RLHFSafety().membership_inference_defense(np.array([0.1, 0.2]))


def test_membership_defense_uses_c_output(monkeypatch):
    lib = Mock()

    def fake_apply(_logits, _count, _epsilon, out):
        ctypes.memmove(out, (ctypes.c_double * 2)(5.0, 6.0), 16)
        return 0

    lib.SNEPPX_membership_inference_defense_apply = Mock(side_effect=fake_apply)
    _use_fake_lib(monkeypatch, lib)

    out = s5.S5RLHFSafety().membership_inference_defense(np.array([0.1, 0.2]))
    assert np.allclose(out, [5.0, 6.0])


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
