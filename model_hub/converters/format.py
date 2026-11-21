"""Format detection utilities."""

from __future__ import annotations

import os
from typing import Optional


def detect_format(path: str) -> str:
    """Detect model format from a file path or directory."""
    if os.path.isdir(path):
        files = os.listdir(path)
        if "config.json" in files and "pytorch_model.bin" in files:
            return "huggingface"
        if "config.json" in files and "sneppx_model.bin" in files:
            return "sneppx-native"
        if "model.safetensors" in files:
            return "safetensors"
        if any(f.endswith((".pt", ".ckpt")) for f in files):
            return "pytorch"
        return "unknown"

    # Check full extension first for ambiguous cases
    lower = path.lower()
    if lower.endswith(".sneppx.bin") or lower.endswith(".sneppx"):
        return "sneppx-native"

    ext = os.path.splitext(path)[1].lower()
    if ext in (".pt", ".ckpt", ".pth"):
        return "pytorch"
    if ext == ".safetensors":
        return "safetensors"
    if ext == ".bin":
        return "huggingface"
    if ext in (".sneppx",):
        return "sneppx-native"
    return "unknown"
