"""HuggingFace ↔ SNEPPX converter."""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict

import numpy as np

from .sneppx_format import load_sneppx_native, save_sneppx_native


def convert_hf_to_sneppx(hf_dir: str, out_dir: str) -> str:
    """Convert a HuggingFace model directory to SNEPPX format."""
    from ..utils import ensure_dir
    ensure_dir(out_dir)

    config_path = os.path.join(hf_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    tensors: Dict[str, Any] = {}
    safetensors_path = os.path.join(hf_dir, "model.safetensors")
    pytorch_path = os.path.join(hf_dir, "pytorch_model.bin")
    hf_model_path = os.path.join(hf_dir, "model.bin")

    if os.path.exists(safetensors_path):
        import safetensors
        with safetensors.safe_open(safetensors_path, framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        source_format = "safetensors"
    elif os.path.exists(pytorch_path) or os.path.exists(hf_model_path):
        import torch
        path = pytorch_path if os.path.exists(pytorch_path) else hf_model_path
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for name, param in checkpoint.items():
            if hasattr(param, "numpy"):
                tensors[name] = param.cpu().numpy()
            else:
                tensors[name] = np.asarray(param)
        source_format = "pytorch"
    else:
        raise FileNotFoundError(f"No model weights found in {hf_dir}")

    out_path = os.path.join(out_dir, "model.sneppx.bin")
    config["_source_format"] = source_format
    config["_hf_path"] = hf_dir
    save_sneppx_native(out_path, tensors, config)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    return out_path


def convert_sneppx_to_hf(sneppx_path: str, out_dir: str) -> str:
    """Convert a SNEPPX native model to HuggingFace format."""
    from ..utils import ensure_dir
    ensure_dir(out_dir)

    meta, tensors = load_sneppx_native(sneppx_path)

    dtype_map = {1: np.float32, 2: np.float16, 3: np.int32, 4: np.int64, 5: np.uint8}
    save_dict = {}

    for tensor_info in meta.get("_tensors", []):
        name = tensor_info["name"]
        shape = tensor_info["shape"]
        dtype_code = tensor_info["dtype_code"]
        data = tensors[name]
        dtype = dtype_map.get(dtype_code, np.float32)
        arr = np.frombuffer(data, dtype=dtype).reshape(shape)
        save_dict[name] = arr

    try:
        from safetensors.numpy import save_file
        out_path = os.path.join(out_dir, "model.safetensors")
        save_file(save_dict, out_path)
    except ImportError:
        import torch
        state_dict = {k: torch.from_numpy(v) for k, v in save_dict.items()}
        out_path = os.path.join(out_dir, "pytorch_model.bin")
        torch.save(state_dict, out_path)

    config = meta.get("_config", meta)
    if "_tensors" in config:
        del config["_tensors"]
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Copy tokenizer files if present
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]:
        src = os.path.join(os.path.dirname(sneppx_path), fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))

    return out_path


def convert_to_hf(source_path: str, out_dir: str, **kwargs) -> str:
    """Convert any supported format to HuggingFace format."""
    from .format import detect_format
    fmt = detect_format(source_path)

    if fmt in ("sneppx-native", "sneppx"):
        return convert_sneppx_to_hf(source_path, out_dir)
    elif fmt == "pytorch":
        return _convert_pt_to_hf(source_path, out_dir, **kwargs)
    elif fmt == "huggingface":
        return _copy_hf_dir(source_path, out_dir)
    elif fmt == "safetensors":
        return _convert_safetensors_to_hf(source_path, out_dir, **kwargs)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def convert_from_hf(hf_dir: str, out_dir: str) -> str:
    """Convert a HuggingFace model to SNEPPX native format."""
    from .format import detect_format
    fmt = detect_format(hf_dir)

    if fmt == "sneppx-native":
        return hf_dir
    elif fmt in ("huggingface", "safetensors"):
        return convert_hf_to_sneppx(hf_dir, out_dir)
    elif fmt == "pytorch":
        return convert_pt_to_sneppx(
            os.path.join(hf_dir, os.listdir(hf_dir)[0] if os.path.isdir(hf_dir) else hf_dir),
            out_dir,
        )[0]  # Handle if directory
    raise ValueError(f"Unsupported format: {fmt}")


def convert_to_sneppx(source_path: str, out_dir: str) -> str:
    """Convert any supported format to SNEPPX native format."""
    from .format import detect_format
    from .pytorch import convert_pt_to_sneppx

    fmt = detect_format(source_path)

    if fmt in ("sneppx-native", "sneppx"):
        # Already in SNEPPX format
        return source_path if os.path.isfile(source_path) else source_path

    if fmt == "huggingface" or fmt == "safetensors":
        return convert_hf_to_sneppx(source_path, out_dir)

    if fmt == "pytorch":
        return convert_pt_to_sneppx(source_path, out_dir)

    raise ValueError(f"Unsupported format: {fmt}")


def convert_from_torch(pt_path: str, out_dir: str) -> str:
    """Convert PyTorch checkpoint to SNEPPX format."""
    from .pytorch import convert_pt_to_sneppx
    return convert_pt_to_sneppx(pt_path, out_dir)


def _convert_pt_to_hf(pt_path: str, out_dir: str, **kwargs) -> str:
    """Convert PyTorch checkpoint directly to HF format."""
    import torch
    from ..utils import ensure_dir
    ensure_dir(out_dir)

    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if config:
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)

    try:
        from safetensors.torch import save_file
        out_path = os.path.join(out_dir, "model.safetensors")
        save_file(state_dict, out_path)
    except ImportError:
        import torch
        out_path = os.path.join(out_dir, "pytorch_model.bin")
        torch.save(state_dict, out_path)

    return out_path


def _copy_hf_dir(src: str, dst: str) -> str:
    """Copy a HuggingFace model directory."""
    from ..utils import ensure_dir
    ensure_dir(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)
    return dst


def _convert_safetensors_to_hf(st_path: str, out_dir: str, **kwargs) -> str:
    """Convert safetensors directly to HF format (just copy)."""
    from ..utils import ensure_dir
    ensure_dir(out_dir)
    shutil.copy2(st_path, os.path.join(out_dir, "model.safetensors"))
    # Copy config if it exists
    config = os.path.join(os.path.dirname(st_path), "config.json")
    if os.path.exists(config):
        shutil.copy2(config, os.path.join(out_dir, "config.json"))
    return os.path.join(out_dir, "model.safetensors")
