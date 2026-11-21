"""HuggingFace Integration — load/save models in HF Transformers format."""

from typing import Dict, List, Optional, Tuple, Union
from .core import Tensor
from .nn import Module, Linear, Embedding, LayerNorm
import json
import os
import numpy as np


def _load_safetensors(path: str) -> Dict[str, np.ndarray]:
    """Load safetensors file (simplified – no safetensors dep)."""
    import struct
    state = {}
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))
        for key, info in header.items():
            offset, size = info['data_offsets']
            cur = f.tell()
            f.seek(8 + header_size + offset)
            data = f.read(size)
            f.seek(cur)
            dtype_str = info['dtype']
            shape = info['shape']
            dt_map = {'F32': np.float32, 'F16': np.float16, 'I32': np.int32, 'I64': np.int64}
            arr = np.frombuffer(data, dtype=dt_map.get(dtype_str, np.float32)).reshape(shape)
            state[key] = arr
    return state


def _load_pytorch_bin(path: str) -> Dict[str, np.ndarray]:
    """Load PyTorch .bin file via numpy."""
    import struct
    state = {}
    with open(path, 'rb') as f:
        data = f.read()
    pos = 0
    while pos < len(data):
        name_len = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
        name = data[pos:pos+name_len].decode('utf-8'); pos += name_len
        dtype_code = struct.unpack('<B', data[pos:pos+1])[0]; pos += 1
        ndim = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
        shape = struct.unpack('<' + 'Q' * ndim, data[pos:pos+8*ndim]); pos += 8 * ndim
        numel = int(np.prod(shape))
        dt_map = {2: np.float32, 3: np.float64, 4: np.float16, 5: np.uint8, 6: np.int32, 7: np.int64}
        dt = dt_map.get(dtype_code, np.float32)
        tensor_bytes = dt(numel).nbytes
        arr = np.frombuffer(data[pos:pos+tensor_bytes], dtype=dt).reshape(shape).copy()
        pos += tensor_bytes
        state[name] = arr
    return state


def load_hf_model(model: Module, model_id: str, cache_dir: Optional[str] = None):
    """Load weights from HuggingFace model."""
    if cache_dir is None:
        cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots")
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
    bin_files = [f for f in os.listdir(snapshot) if f.endswith(('.bin', '.safetensors'))]
    if not bin_files:
        print(f"No weight files found in {snapshot}")
        return
    state = {}
    for fname in bin_files:
        path = os.path.join(snapshot, fname)
        try:
            if fname.endswith('.safetensors'):
                state.update(_load_safetensors(path))
            elif fname.endswith('.bin'):
                state.update(_load_pytorch_bin(path))
        except Exception as e:
            print(f"  Warning: could not load {fname}: {e}")
    if not state:
        return
    hf_to_sneppx = {}
    for name, param in model.named_parameters():
        hf_name = name.replace('_modules.', '').replace('_parameters.', '')
        hf_to_sneppx[hf_name] = name
    loaded = 0
    for hf_name, arr in state.items():
        mapped = hf_to_sneppx.get(hf_name)
        if mapped:
            param = dict(model.named_parameters())[mapped]
            if param.shape == arr.shape:
                param._data._data[:] = arr.tobytes()
                loaded += 1
    print(f"Loaded {loaded}/{len(list(model.parameters()))} parameters from {model_id}")


def save_hf_model(model: Module, save_dir: str, model_id: str = "sneppx-model"):
    """Save model in HF-compatible safetensors format."""
    os.makedirs(save_dir, exist_ok=True)
    state = {}
    for name, param in model.named_parameters():
        clean_name = name.replace('_modules.', '').replace('_parameters.', '')
        arr = param.numpy()
        state[clean_name] = arr
    header = {}
    offset = 0
    data = b''
    for name, arr in state.items():
        numel = int(np.prod(arr.shape))
        nbytes = numel * arr.itemsize
        dt_map = {np.float32: 'F32', np.float16: 'F16', np.int32: 'I32', np.int64: 'I64'}
        dtype_str = dt_map.get(arr.dtype.type, 'F32')
        header[name] = {'dtype': dtype_str, 'shape': list(arr.shape), 'data_offsets': [offset, offset + nbytes]}
        data += arr.tobytes()
        offset += nbytes
    header_bytes = json.dumps(header).encode('utf-8')
    with open(os.path.join(save_dir, "model.safetensors"), 'wb') as f:
        f.write(len(header_bytes).to_bytes(8, 'little'))
        f.write(header_bytes)
        f.write(data)
    config = {
        "model_id": model_id,
        "architectures": ["SneppxForCausalLM"],
        "model_type": "sneppx",
    }
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved model to {save_dir}/")


def load_config(config_path: str) -> Dict:
    """Load model config from HF config.json."""
    with open(config_path, 'r') as f:
        return json.load(f)