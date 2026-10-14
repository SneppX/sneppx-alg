"""HuggingFace Integration — load/save models in HF Transformers format."""

from typing import Dict, List, Optional, Tuple, Union
from .tensor import Tensor
from .nn import Module, Linear, Embedding, LayerNorm
import json
import os
import numpy as np


def _decode_bfloat16(raw: bytes) -> np.ndarray:
    """Decode little-endian bfloat16 bytes into a float32 ndarray.

    bfloat16 is the top 16 bits of an IEEE-754 float32, so widening is a
    shift-left by 16 bits into a uint32 buffer reinterpreted as float32.
    """
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def _load_safetensors(path: str) -> Dict[str, np.ndarray]:
    import struct

    state = {}
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        for key, info in header.items():
            # safetensors may carry a top-level "__metadata__" entry whose
            # value is a string map, not a tensor descriptor — skip it.
            if not isinstance(info, dict) or "data_offsets" not in info:
                continue
            offset, size = info["data_offsets"]
            cur = f.tell()
            f.seek(8 + header_size + offset)
            data = f.read(size)
            f.seek(cur)
            dtype_str = info["dtype"]
            shape = info["shape"]
            if dtype_str == "BF16":
                arr = _decode_bfloat16(data).reshape(shape)
            else:
                dt_map = {
                    "F32": np.float32,
                    "F16": np.float16,
                    "F64": np.float64,
                    "I32": np.int32,
                    "I64": np.int64,
                    "I16": np.int16,
                    "I8": np.int8,
                    "U8": np.uint8,
                    "BOOL": np.bool_,
                }
                arr = np.frombuffer(
                    data, dtype=dt_map.get(dtype_str, np.float32)
                ).reshape(shape)
            state[key] = arr
    return state


def _load_pytorch_bin(path: str) -> Dict[str, np.ndarray]:
    import struct

    state = {}
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    while pos < len(data):
        name_len = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4
        name = data[pos : pos + name_len].decode("utf-8")
        pos += name_len
        dtype_code = struct.unpack("<B", data[pos : pos + 1])[0]
        pos += 1
        ndim = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4
        shape = struct.unpack("<" + "Q" * ndim, data[pos : pos + 8 * ndim])
        pos += 8 * ndim
        numel = int(np.prod(shape))
        dt_map = {
            2: np.float32,
            3: np.float64,
            4: np.float16,
            5: np.uint8,
            6: np.int32,
            7: np.int64,
        }
        dt = dt_map.get(dtype_code, np.float32)
        tensor_bytes = dt(numel).nbytes
        arr = (
            np.frombuffer(data[pos : pos + tensor_bytes], dtype=dt)
            .reshape(shape)
            .copy()
        )
        pos += tensor_bytes
        state[name] = arr
    return state


def _named_parameters(model) -> List[Tuple[str, Tensor]]:
    """Iterate (name, param) pairs for any module.

    Modules with ``named_parameters()`` (``nn.Module`` and friends) are used
    directly; others (e.g. ``keras_api.Sequential``) only expose a flat
    ``parameters()`` list, which is named ``param_<i>``.
    """
    if hasattr(model, "named_parameters"):
        return list(model.named_parameters())
    return [(f"param_{i}", p) for i, p in enumerate(model.parameters())]


def load_hf_model(model: Module, model_id: str, cache_dir: Optional[str] = None):
    if cache_dir is None:
        cache_dir = os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots"
        )
    if not os.path.exists(cache_dir):
        print(f"Cache directory not found: {cache_dir}")
        print("Download the model first using huggingface_hub:")
        print(f"  from huggingface_hub import snapshot_download")
        print(f"  snapshot_download('{model_id}')")
        return
    snapshots = os.listdir(cache_dir)
    if not snapshots:
        return
    snapshot = os.path.join(cache_dir, snapshots[0])
    bin_files = [
        f for f in os.listdir(snapshot) if f.endswith((".bin", ".safetensors"))
    ]
    if not bin_files:
        print(f"No weight files found in {snapshot}")
        return
    state = {}
    for fname in bin_files:
        path = os.path.join(snapshot, fname)
        try:
            if fname.endswith(".safetensors"):
                state.update(_load_safetensors(path))
            elif fname.endswith(".bin"):
                state.update(_load_pytorch_bin(path))
        except Exception as e:
            print(f"  Warning: could not load {fname}: {e}")
    if not state:
        return
    hf_to_sneppx = {}
    for name, _ in _named_parameters(model):
        hf_name = name.replace("_modules.", "").replace("_parameters.", "")
        hf_to_sneppx[hf_name] = name
    named = dict(_named_parameters(model))
    loaded = 0
    for hf_name, arr in state.items():
        mapped = hf_to_sneppx.get(hf_name)
        if mapped:
            param = named[mapped]
            if param.shape == arr.shape:
                arr_cont = np.ascontiguousarray(arr)
                param.data = arr_cont
                loaded += 1
    print(f"Loaded {loaded}/{len(list(model.parameters()))} parameters from {model_id}")


def save_hf_model(model: Module, save_dir: str, model_id: str = "sneppx-model"):
    os.makedirs(save_dir, exist_ok=True)
    state = {}
    for name, param in _named_parameters(model):
        clean_name = name.replace("_modules.", "").replace("_parameters.", "")
        arr = param.data
        state[clean_name] = arr
    header = {}
    offset = 0
    data = b""
    for name, arr in state.items():
        numel = int(np.prod(arr.shape))
        nbytes = numel * arr.itemsize
        dt_map = {
            np.float32: "F32",
            np.float16: "F16",
            np.int32: "I32",
            np.int64: "I64",
        }
        dtype_str = dt_map.get(arr.dtype.type, "F32")
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        data += arr.tobytes()
        offset += nbytes
    header_bytes = json.dumps(header).encode("utf-8")
    with open(os.path.join(save_dir, "model.safetensors"), "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(data)
    config = {
        "model_id": model_id,
        "architectures": ["SneppxForCausalLM"],
        "model_type": "sneppx",
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved model to {save_dir}/")


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return json.load(f)


# Model-id -> (family, size) patterns supported by from_pretrained().
_MODEL_ID_PATTERNS = [
    ("llama-2-7b", "llama2", "7B"),
    ("llama-2-13b", "llama2", "13B"),
    ("llama-2-70b", "llama2", "70B"),
    ("llama-3-8b", "llama3", "8B"),
    ("llama-3-70b", "llama3", "70B"),
    ("mistral-7b", "mistral", "7B"),
    ("qwen2-7b", "qwen2", "7B"),
    ("qwen2-72b", "qwen2", "72B"),
    ("deepseek-v2-lite", "deepseek_v2", "lite"),
    ("deepseek-v2", "deepseek_v2", "full"),
]


def _resolve_family_size(model_id: str):
    lower = model_id.lower()
    for pattern, family, size in _MODEL_ID_PATTERNS:
        if pattern in lower:
            return family, size
    raise ValueError(
        f"Unknown model_id '{model_id}'. "
        f"Supported: {[p[0] for p in _MODEL_ID_PATTERNS]}"
    )


def _find_snapshot(model_id: str, cache_dir: Optional[str]) -> Optional[str]:
    """Locate an existing snapshot directory, downloading it if possible.

    Returns the snapshot path or None if no local weights are available.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots"
        )
    if os.path.isdir(cache_dir) and os.listdir(cache_dir):
        return cache_dir if os.listdir(cache_dir)[0] else cache_dir

    # Try to download via huggingface_hub if available.
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        local = snapshot_download(model_id, cache_dir=cache_dir)
        return local
    except Exception as e:  # network/unavailable — fall back to None
        print(f"[SNEPPX from_pretrained] could not download '{model_id}': {e}")
        return None


def from_pretrained(
    model_id: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    verbose: bool = True,
):
    """Load a pretrained LLM from HuggingFace and build a SneppX model.

    Supported families: LLaMA-2/3, Mistral, Qwen2, DeepSeek-V2.

    Args:
        model_id: HF-style model ID (e.g. ``meta-llama/Llama-2-7b-hf``).
        cache_dir: Local snapshot/cache directory. If omitted, the canonical
            HuggingFace cache location is used and a download is attempted.
        force_download: Re-download even if a local snapshot exists.
        verbose: Print progress.

    Returns:
        A :class:`~SneppX_ALG.interface_bindings.nn.Transformer` with weights
        loaded from the HF checkpoint (or an uninitialized model if no weights
        could be located).
    """
    from .model_zoo import (
        get_model_config,
        build_transformer_from_config,
        _remap_hf_weight_name,
    )

    family, size = _resolve_family_size(model_id)
    config = get_model_config(family, size)
    model = build_transformer_from_config(config)

    if force_download:
        cache_dir = None
    snapshot = _find_snapshot(model_id, cache_dir)
    if snapshot is None or not os.listdir(snapshot):
        if verbose:
            print(
                f"[SNEPPX from_pretrained] no local weights for '{model_id}'; "
                "returning uninitialized model."
            )
        return model

    # Gather weights from every safetensors/bin file in the snapshot.
    state: Dict[str, np.ndarray] = {}
    for fname in sorted(os.listdir(snapshot)):
        path = os.path.join(snapshot, fname)
        try:
            if fname.endswith(".safetensors"):
                state.update(_load_safetensors(path))
            elif fname.endswith(".bin"):
                state.update(_load_pytorch_bin(path))
        except Exception as e:
            if verbose:
                print(f"  Warning: could not load {fname}: {e}")

    if not state:
        if verbose:
            print(f"[SNEPPX from_pretrained] no weight files in {snapshot}")
        return model

    params = dict(model.named_parameters())
    loaded = 0
    for hf_name, arr in state.items():
        sneppx_name = _remap_hf_weight_name(hf_name, family)
        if sneppx_name and sneppx_name in params:
            param = params[sneppx_name]
            if param.shape == arr.shape:
                param.data = np.ascontiguousarray(arr, dtype=param.data.dtype)
                loaded += 1
            elif verbose:
                print(
                    f"  shape mismatch {sneppx_name}: "
                    f"model {param.shape} vs weight {arr.shape}"
                )
    if verbose:
        print(
            f"[SNEPPX from_pretrained] loaded {loaded}/"
            f"{len(params)} params from {model_id} ({family}/{size})"
        )
    return model
