"""FP8/INT4 quantized model serving — wraps QuantizedLinear into serving-ready format."""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

import numpy as np

from .tensor import Tensor
from .quantization import QuantMode, QuantGranularity, QuantizedLinear
from .nn import Module, Linear


@dataclass
class QuantizedModelConfig:
    quant_mode: int = QuantMode.INT4_SYM
    skip_layers: List[str] = field(default_factory=lambda: ["lm_head", "embed_tokens"])


def quantize_model_weights(
    model_params: Dict[str, np.ndarray],
    config: Optional[QuantizedModelConfig] = None,
) -> Dict[str, Any]:
    cfg = config or QuantizedModelConfig()
    quantized = {}
    for name, weight in model_params.items():
        skip = any(s in name for s in cfg.skip_layers)
        if skip:
            quantized[name] = weight
            continue
        w_t = Tensor.from_numpy(weight)
        if cfg.quant_mode == QuantMode.INT4_SYM:
            from .quantization import quantize_int4_sym
            qw, scale = quantize_int4_sym(w_t)
            ql = QuantizedLinear(qw, scale, mode=cfg.quant_mode)
        else:
            from .quantization import quantize_int8_sym
            qw, scale = quantize_int8_sym(w_t)
            ql = QuantizedLinear(qw, scale, mode=cfg.quant_mode)
        quantized[name] = ql
    return quantized


def dequantize_weights(
    quantized: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    result = {}
    for name, item in quantized.items():
        if isinstance(item, QuantizedLinear):
            w = item.weight.data if hasattr(item.weight, "data") else item.weight
            result[name] = np.asarray(w, dtype=np.float32)
        elif isinstance(item, np.ndarray):
            result[name] = item
    return result


def quantized_forward(
    quantized_params: Dict[str, Any],
    layer_name: str,
    x: np.ndarray,
) -> np.ndarray:
    item = quantized_params.get(layer_name)
    if isinstance(item, QuantizedLinear):
        x_t = Tensor.from_numpy(x)
        out_t = item(x_t)
        return out_t.data if hasattr(out_t, "data") else np.asarray(out_t)
    elif isinstance(item, np.ndarray):
        return x @ item.T
    return x


def estimate_model_size_mb(quantized: Dict[str, Any]) -> float:
    total_bytes = 0
    for name, item in quantized.items():
        if isinstance(item, QuantizedLinear):
            w_data = item.weight.data if hasattr(item.weight, "data") else item.weight
            total_bytes += np.asarray(w_data).nbytes
        elif isinstance(item, np.ndarray):
            total_bytes += item.nbytes
    return total_bytes / (1024 * 1024)
