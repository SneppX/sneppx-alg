"""Regression: model converters must not unpickle arbitrary PyTorch globals."""

import os
import sys
from types import SimpleNamespace

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from model_hub.converters import pytorch as pt_converter


def test_safe_torch_load_enables_weights_only(monkeypatch):
    calls = {}

    def fake_load(path, **kwargs):
        calls.update(kwargs)
        return {"weight": object()}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=fake_load))
    pt_converter.safe_torch_load("model.pt")
    assert calls == {"map_location": "cpu", "weights_only": True}


def test_converters_have_no_weights_only_false_call():
    converters_dir = os.path.join(ROOT, "model_hub", "converters")
    for name in ("pytorch.py", "huggingface.py"):
        with open(os.path.join(converters_dir, name), "r", encoding="utf-8") as handle:
            assert "weights_only=False" not in handle.read()
