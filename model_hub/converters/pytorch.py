"""PyTorch ↔ SNEPPX converter."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .format import detect_format
from .sneppx_format import load_sneppx_native, save_sneppx_native


def convert_pt_to_sneppx(pt_path: str, out_dir: str) -> str:
    """Convert a PyTorch checkpoint to SNEPPX native format."""
    import torch
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    elif hasattr(checkpoint, "state_dict"):
        state_dict = checkpoint.state_dict()
    else:
        raise ValueError("Unrecognized checkpoint format")

    numpy_tensors: Dict[str, Any] = {}
    for name, param in state_dict.items():
        if hasattr(param, "data"):
            arr = param.data.cpu().numpy()
        elif hasattr(param, "numpy"):
            arr = param.cpu().numpy()
        else:
            arr = param
        numpy_tensors[name] = np.ascontiguousarray(arr)

    out_path = os.path.join(out_dir, "model.sneppx.bin")
    save_sneppx_native(out_path, numpy_tensors, {
        "source_format": "pytorch",
        "num_tensors": len(numpy_tensors),
        "original_path": pt_path,
    })

    if isinstance(checkpoint, dict) and "config" in checkpoint:
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(checkpoint["config"], f, indent=2, default=str)

    return out_path


def convert_sneppx_to_pt(sneppx_path: str, out_path: str, metadata: Optional[Dict] = None) -> str:
    """Convert a SNEPPX native model to PyTorch format."""
    import torch
    import numpy as np

    meta, tensors = load_sneppx_native(sneppx_path)

    dtype_map = {1: np.float32, 2: np.float16, 3: np.int32, 4: np.int64, 5: np.uint8}
    state_dict = {}

    for tensor_info in meta.get("_tensors", []):
        name = tensor_info["name"]
        shape = tensor_info["shape"]
        dtype_code = tensor_info["dtype_code"]
        data = tensors[name]
        dtype = dtype_map.get(dtype_code, np.float32)
        arr = np.frombuffer(data, dtype=dtype).reshape(shape)
        state_dict[name] = torch.from_numpy(arr)

    config = metadata or meta.get("_config", {})
    checkpoint = {"model_state_dict": state_dict, "config": config}
    torch.save(checkpoint, out_path)

    return out_path
