"""SNEPPX native binary format read/write."""

from __future__ import annotations

import json
import struct
from typing import Any, Dict, Tuple

SNEPPX_MAGIC = b"SNEPPX0001"
SNEPPX_VERSION = 1


def load_sneppx_native(path: str) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    """Load a SNEPPX native model file.

    Format:
      - 9 bytes: magic ("SNEPPX0001")
      - 4 bytes: version (uint32, big-endian)
      - 4 bytes: metadata size (uint32, big-endian)
      - metadata_size bytes: JSON metadata
      - 4 bytes: number of tensors
      - For each tensor:
        - 4 bytes: name length (uint32)
        - N bytes: name (UTF-8)
        - 4 bytes: shape length (uint32)
        - 4 * shape_len bytes: shape dims (uint32 each, big-endian)
        - 4 bytes: dtype code (uint32: 1=f32, 2=f16, 3=i32, 4=i64, 5=u8)
        - 8 bytes: data size (uint64, big-endian)
        - data_size bytes: raw tensor data
    """
    with open(path, "rb") as f:
        magic = f.read(10)
        if magic != SNEPPX_MAGIC:
            raise ValueError(f"Not a SNEPPX native file: {path}")

        version = struct.unpack(">I", f.read(4))[0]
        if version != SNEPPX_VERSION:
            raise ValueError(f"Unsupported SNEPPX format version: {version}")

        meta_size = struct.unpack(">I", f.read(4))[0]
        metadata = json.loads(f.read(meta_size).decode("utf-8")) if meta_size > 0 else {}

        num_tensors = struct.unpack(">I", f.read(4))[0]

        tensors: Dict[str, bytes] = {}
        tensor_meta: list = []

        for _ in range(num_tensors):
            name_len = struct.unpack(">I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")

            shape_len = struct.unpack(">I", f.read(4))[0]
            shape = list(struct.unpack(f">{shape_len}I", f.read(4 * shape_len)))

            dtype_code = struct.unpack(">I", f.read(4))[0]
            data_size = struct.unpack(">Q", f.read(8))[0]
            data = f.read(data_size)

            tensors[name] = data
            tensor_meta.append({
                "name": name,
                "shape": shape,
                "dtype_code": dtype_code,
                "byte_size": data_size,
            })

        metadata["_tensors"] = tensor_meta
        return metadata, tensors


def save_sneppx_native(path: str, tensors: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save tensors in SNEPPX native format.

    Format:
      - 9 bytes: magic ("SNEPPX0001")
      - 4 bytes: version (uint32, big-endian)
      - 4 bytes: metadata size (uint32, big-endian)
      - metadata_size bytes: JSON metadata
      - 4 bytes: number of tensors
      - For each tensor:
        - 4 bytes: name length (uint32)
        - N bytes: name (UTF-8)
        - 4 bytes: shape length (uint32)
        - 4 * shape_len bytes: shape dims (uint32 each, big-endian)
        - 4 bytes: dtype code (uint32: 1=f32, 2=f16, 3=i32, 4=i64, 5=u8)
        - 8 bytes: data size (uint64, big-endian)
        - data_size bytes: raw tensor data
    """
    import numpy as np

    dtype_map = {
        np.float32: 1,
        np.float16: 2,
        np.int32: 3,
        np.int64: 4,
        np.uint8: 5,
    }

    meta = metadata or {}
    tensor_meta = []

    # Build tensor metadata for JSON
    for name, arr in tensors.items():
        arr_check = np.ascontiguousarray(arr)
        dtype_code = dtype_map.get(arr_check.dtype.type, 1)
        tensor_meta.append({
            "name": name,
            "shape": list(arr_check.shape),
            "dtype_code": dtype_code,
            "byte_size": arr_check.nbytes,
        })

    meta["_tensors"] = tensor_meta

    # Serialize metadata first
    import io
    meta_json = json.dumps(meta)
    meta_bytes = meta_json.encode("utf-8")

    with open(path, "wb") as f:
        f.write(SNEPPX_MAGIC)
        f.write(struct.pack(">I", SNEPPX_VERSION))
        # Write metadata size + metadata
        f.write(struct.pack(">I", len(meta_bytes)))
        f.write(meta_bytes)
        # Write tensor count
        f.write(struct.pack(">I", len(tensors)))
        # Write tensor data
        for name, arr in tensors.items():
            arr = np.ascontiguousarray(arr)
            name_bytes = name.encode("utf-8")
            shape = list(arr.shape)
            dtype_code = dtype_map.get(arr.dtype.type, 1)

            f.write(struct.pack(">I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack(">I", len(shape)))
            for dim in shape:
                f.write(struct.pack(">I", dim))
            f.write(struct.pack(">I", dtype_code))
            f.write(struct.pack(">Q", arr.nbytes))
            f.write(arr.tobytes())
