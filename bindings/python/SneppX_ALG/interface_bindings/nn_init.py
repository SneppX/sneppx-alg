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
    if len(arr.shape) == 0:
        return 1, 1
    if len(arr.shape) == 1:
        return arr.shape[0], arr.shape[0]
    if len(arr.shape) == 2:
        return arr.shape[1], arr.shape[0]
    # Convolution-like: shape = (out_channels, in_channels, *spatial)
    receptive_field_size = 1
    for d in arr.shape[2:]:
        receptive_field_size *= d
    fan_in = arr.shape[1] * receptive_field_size
    fan_out = arr.shape[0] * receptive_field_size
    return fan_in, fan_out


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    arr = _np(tensor)
    _write(tensor, np.random.uniform(-bound, bound, size=arr.shape))
    return tensor


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    arr = _np(tensor)
    _write(tensor, np.random.normal(0.0, std, size=arr.shape))
    return tensor


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity, a)
    bound = gain * math.sqrt(3.0 / max(fan, 1))
    arr = _np(tensor)
    _write(tensor, np.random.uniform(-bound, bound, size=arr.shape))
    return tensor


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(max(fan, 1))
    arr = _np(tensor)
    _write(tensor, np.random.normal(0.0, std, size=arr.shape))
    return tensor


def trunc_normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Tensor:
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


def eye_(tensor: Tensor) -> Tensor:
    """Fill the 2D tensor with the identity matrix."""
    arr = _np(tensor)
    if len(arr.shape) != 2:
        raise ValueError("eye_ expects a 2D tensor")
    _write(tensor, np.eye(arr.shape[0], arr.shape[1]))
    return tensor


def orthogonal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill the tensor with an (approximately) orthogonal matrix.

    For 2D tensors a QR decomposition of a random Gaussian matrix is used;
    for higher-order tensors the first two dimensions are orthogonalised.
    """
    arr = _np(tensor)
    rows = arr.shape[0]
    cols = arr.shape[1] if len(arr.shape) >= 2 else 1
    flat = np.random.normal(0.0, 1.0, size=(rows, cols)).astype(arr.dtype)
    q, r = np.linalg.qr(flat)
    # Enforce a deterministic sign convention
    q *= np.sign(np.diag(r))
    q = q * gain
    if len(arr.shape) == 1:
        out = q.reshape(arr.shape)
    else:
        out = q.reshape((rows, cols) + tuple(arr.shape[2:]))
    _write(tensor, out.astype(arr.dtype))
    return tensor
