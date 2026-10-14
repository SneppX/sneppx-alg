"""Tests for the safetensors loader in hf_integration.py.

Covers BF16 decoding and the ``__metadata__`` header key that real
HuggingFace safetensors files carry.
"""

import json
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python")
)

from SneppX_ALG.interface_bindings.hf_integration import (
    _load_safetensors,
    _decode_bfloat16,
)


def _write_safetensors(path, tensors, metadata=None):
    """Write a minimal safetensors file from {name: (dtype_str, ndarray)}."""
    header = {}
    offset = 0
    blob = b""
    if metadata is not None:
        header["__metadata__"] = metadata
    for name, (dtype_str, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blob += raw
        offset += len(raw)
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(blob)


def _bfloat16_bytes(arr_f32):
    u32 = np.asarray(arr_f32, dtype=np.float32).view(np.uint32)
    return (u32 >> np.uint32(16)).astype("<u2").tobytes()


def test_decode_bfloat16_roundtrip():
    vals = np.array([1.0, 2.0, -0.5, 4.0, 0.0], dtype=np.float32)
    decoded = _decode_bfloat16(_bfloat16_bytes(vals))
    np.testing.assert_array_equal(decoded, vals)
    print("  test_decode_bfloat16_roundtrip PASS")


def test_load_safetensors_skips_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.safetensors")
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        _write_safetensors(
            path,
            {"weight": ("F32", arr)},
            metadata={"format": "pt", "author": "test"},
        )
        state = _load_safetensors(path)
        assert list(state.keys()) == ["weight"]
        np.testing.assert_array_equal(state["weight"], arr)
    print("  test_load_safetensors_skips_metadata PASS")


def test_load_safetensors_bfloat16():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.safetensors")
        vals = np.array([1.0, 2.0, -0.5, 4.0], dtype=np.float32)
        raw = _bfloat16_bytes(vals)
        header = {"w": {"dtype": "BF16", "shape": [4], "data_offsets": [0, len(raw)]}}
        header_bytes = json.dumps(header).encode("utf-8")
        with open(path, "wb") as f:
            f.write(len(header_bytes).to_bytes(8, "little"))
            f.write(header_bytes)
            f.write(raw)
        state = _load_safetensors(path)
        assert state["w"].dtype == np.float32
        np.testing.assert_array_equal(state["w"], vals)
    print("  test_load_safetensors_bfloat16 PASS")


if __name__ == "__main__":
    test_decode_bfloat16_roundtrip()
    test_load_safetensors_skips_metadata()
    test_load_safetensors_bfloat16()
    print("\nAll safetensors loader tests passed!")
