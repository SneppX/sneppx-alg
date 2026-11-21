"""Parameter initialization utilities (torch.nn.init-compatible API).

All functions mutate the passed Tensor in place and return it, matching
``torch.nn.init`` semantics.
"""

from typing import Optional
import math
import numpy as np

from .tensor import Tensor


def _np(t: Tensor) -> np.ndarray:
    return np.asarray(t.data, dtype=np.float64)


def _write(t: Tensor, arr: np.ndarray) -> Tensor:
    t.data = arr.astype(t.data.dtype if hasattr(t.data, "dtype") else arr.dtype)
    return t


def zeros_(tensor: Tensor) -> Tensor:
    _write(tensor, np.zeros_like(_np(tensor)))
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    _write(tensor, np.ones_like(_np(tensor)))
    return tensor


def constant_(tensor: Tensor, val: float) -> Tensor:
    _write(tensor, np.full_like(_np(tensor), val))
    return tensor


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    arr = _np(tensor)
    _write(tensor, np.random.uniform(a, b, size=arr.shape))
    return tensor


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    arr = _np(tensor)
    _write(tensor, np.random.normal(mean, std, size=arr.shape))
    return tensor


def _calculate_fan_in_and_fan_out(tensor: Tensor):
    arr = _np(tensor)
    receptive_field_size = 1
    for d in arr.shape[2:]:
        receptive_field_size *= d
    fan_in = arr.shape[1] * receptive_field_size
    fan_out = arr.shape[0] * receptive_field_size
    return fan_in, fan_out


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    arr = _np(tensor)
    _write(tensor, np.random.uniform(-a, a, size=arr.shape))
    return tensor


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    arr = _np(tensor)
    _write(tensor, np.random.normal(0.0, std, size=arr.shape))
    return tensor


def kaiming_uniform_(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = math.sqrt(2.0 / (1 + a * a)) if nonlinearity == "leaky_relu" else 1.0
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    arr = _np(tensor)
    _write(tensor, np.random.uniform(-bound, bound, size=arr.shape))
    return tensor


def kaiming_normal_(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = math.sqrt(2.0 / (1 + a * a)) if nonlinearity == "leaky_relu" else 1.0
    std = gain / math.sqrt(fan)
    arr = _np(tensor)
    _write(tensor, np.random.normal(0.0, std, size=arr.shape))
    return tensor


def trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> Tensor:
    arr = _np(tensor)
    out = np.empty(arr.shape)
    n = 0
    while n < out.size:
        sample = np.random.normal(mean, std)
        if a <= (sample - mean) / std <= b:
            out.flat[n] = sample
            n += 1
    _write(tensor, out)
    return tensor


def calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    linear = 1.0
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        return math.sqrt(2.0 / (1 + (param or 0.01) ** 2))
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity == "sigmoid":
        return 1.0
    return linear
