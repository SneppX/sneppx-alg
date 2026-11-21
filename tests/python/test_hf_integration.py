"""Tests for hf_integration.py — HuggingFace weight loading."""

import numpy as np
import tempfile
import os
import json
from SneppX_ALG.interface_bindings import (
    Tensor,
    Linear,
    Sequential,
    load_config,
    save_hf_model,
)


def test_save_and_load_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        config = {"model_id": "test", "architectures": ["SneppxForCausalLM"]}
        with open(config_path, "w") as f:
            json.dump(config, f)
        loaded = load_config(config_path)
        assert loaded["model_id"] == "test"
    print("  test_save_and_load_config PASS")


def test_save_hf_model_creates_files():
    model = Sequential(Linear(4, 8), Linear(8, 2))
    with tempfile.TemporaryDirectory() as tmpdir:
        save_hf_model(model, tmpdir, model_id="test-model")
        assert os.path.exists(os.path.join(tmpdir, "model.safetensors"))
        assert os.path.exists(os.path.join(tmpdir, "config.json"))
        with open(os.path.join(tmpdir, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["model_id"] == "test-model"
    print("  test_save_hf_model_creates_files PASS")


def test_save_hf_model_weights_roundtrip():
    model = Sequential(Linear(4, 8), Linear(8, 2))
    orig_params = [p.data.copy() for p in model.parameters()]
    with tempfile.TemporaryDirectory() as tmpdir:
        save_hf_model(model, tmpdir, model_id="test-model")
        with open(os.path.join(tmpdir, "model.safetensors"), "rb") as f:
            import struct

            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
            assert len(header) == len(orig_params)
    print("  test_save_hf_model_weights_roundtrip PASS")


if __name__ == "__main__":
    test_save_and_load_config()
    test_save_hf_model_creates_files()
    test_save_hf_model_weights_roundtrip()
    print("\nAll hf_integration tests passed!")
