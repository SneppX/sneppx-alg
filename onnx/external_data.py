"""ONNX external-data support (numpy-only).

Saves initializer payloads to separate binary files and rewrites the model's
TensorProtos to reference them via ``data_location=EXTERNAL`` + external_data
entries (location / offset / length), and loads them back on parse.
"""

import os
from typing import Dict, Optional

import numpy as np

from .model import Model, Tensor
from .parser import load_model, parse_model

__all__ = ["save_external_data", "extract_external_data", "load_external_data"]


def _to_bytes(data: np.ndarray) -> bytes:
    raw = np.ascontiguousarray(data)
    if raw.dtype.byteorder == ">":
        raw = raw.astype(raw.dtype.newbyteorder("<"))
    return raw.tobytes()


def save_external_data(
    model: Model,
    output_dir: str,
    size_threshold: int = 1024,
    location: str = "weights.bin",
) -> Model:
    """Move initializers larger than ``size_threshold`` bytes to external files.

    Returns a new :class:`Model` whose heavy initializers carry an
    ``external_data`` dict (location/offset/length) instead of inlined data.
    """
    os.makedirs(output_dir, exist_ok=True)
    bin_path = os.path.join(output_dir, location)
    offset = 0
    with open(bin_path, "wb") as f:
        for init in model.graph.initializers:
            if init.data is None or init.data.nbytes <= size_threshold:
                continue
            payload = _to_bytes(init.data)
            f.write(payload)
            init.external_data = {
                "location": location,
                "offset": str(offset),
                "length": str(len(payload)),
            }
            init.data_location = "EXTERNAL"
            init.data = None
            offset += len(payload)
    return model


def extract_external_data(
    model: Model, base_dir: Optional[str] = None
) -> Model:
    """Load external initializer payloads back into ``tensor.data``.

    ``base_dir`` defaults to the directory containing the model file (callers
    should pass ``os.path.dirname(path)`` when loading from a file).
    """
    for init in model.graph.initializers:
        if not init.external_data:
            continue
        location = init.external_data.get("location", "")
        offset = int(init.external_data.get("offset", 0))
        length = int(init.external_data.get("length", 0))
        path = location if os.path.isabs(location) else os.path.join(
            base_dir or "", location
        )
        with open(path, "rb") as f:
            f.seek(offset)
            payload = f.read(length)
        nptype = np.dtype(
            {
                "float32": np.float32, "float64": np.float64, "float16": np.float16,
                "int8": np.int8, "uint8": np.uint8, "int16": np.int16,
                "uint16": np.uint16, "int32": np.int32, "uint32": np.uint32,
                "int64": np.int64, "uint64": np.uint64, "bool": np.bool_,
            }.get(init.dtype, np.float32)
        )
        arr = np.frombuffer(payload, dtype=nptype)
        init.data = arr.reshape(init.shape) if init.shape else arr
        init.external_data = {}
        init.data_location = "DEFAULT"
    return model


def load_external_data(path: str, base_dir: Optional[str] = None) -> Model:
    """Load a model file and hydrate any external data references."""
    model = load_model(path)
    return extract_external_data(model, base_dir or os.path.dirname(path))
