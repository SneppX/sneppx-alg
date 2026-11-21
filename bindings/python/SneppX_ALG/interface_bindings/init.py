"""Parameter initialization utilities (in-place Tensor fillers).

Mirrors the torch.nn.init surface but operating on SneppX_ALG Tensors.

Typical usage:
    import SneppX_ALG.interface_bindings.init as init
    init.xavier_uniform_(linear.weight)
    init.zeros_(linear.bias)
"""

import math
import numpy as np

from .tensor import Tensor


# ---------------------------------------------------------------------------
# Fan-in / fan-out computation (PyTorch convention)
# ---------------------------------------------------------------------------


def _calculate_fan_in_and_fan_out(shape):
    """Return (fan_in, fan_out) for a parameter tensor of the given shape."""
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        return shape[0], shape[0]
    if len(shape) == 2:
        return shape[1], shape[0]

    # Convolution-like: shape = (out_channels, in_channels, *spatial)
    receptive_field = 1
    for s in shape[2:]:
        receptive_field *= s
    fan_in = shape[1] * receptive_field
    fan_out = shape[0] * receptive_field
    return fan_in, fan_out


def calculate_gain(nonlinearity, param=None):
    """Return the recommended gain for the given nonlinearity (PyTorch semantics)."""
    nl = (nonlinearity or "linear").lower()
    if nl in ("linear", "conv1d", "conv2d", "conv3d",
              "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
              "sigmoid"):
        return 1.0
    if nl == "tanh":
        return 5.0 / 3.0
    if nl == "relu":
        return math.sqrt(2.0)
    if nl == "leaky_relu":
        a = param if param is not None else 0.01
        return math.sqrt(2.0 / (1.0 + a * a))
    if nl == "selu":
        return 3.0 / 4.0
    return 1.0


# ---------------------------------------------------------------------------
# Basic fillers
# ---------------------------------------------------------------------------


def zeros_(tensor: Tensor) -> Tensor:
    tensor.data = np.zeros(tensor.shape, dtype=tensor.data.dtype)
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    tensor.data = np.ones(tensor.shape, dtype=tensor.data.dtype)
    return tensor


def constant_(tensor: Tensor, val: float) -> Tensor:
    tensor.data = np.full(tensor.shape, val, dtype=tensor.data.dtype)
    return tensor


def eye_(tensor: Tensor) -> Tensor:
    if len(tensor.shape) != 2:
        raise ValueError("eye_ expects a 2D tensor")
    arr = np.eye(tensor.shape[0], tensor.shape[1], dtype=tensor.data.dtype)
    tensor.data = arr
    return tensor


# ---------------------------------------------------------------------------
# Random fillers
# ---------------------------------------------------------------------------


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    arr = np.random.uniform(a, b, size=tensor.shape).astype(tensor.data.dtype)
    tensor.data = arr
    return tensor


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    arr = np.random.normal(mean, std, size=tensor.shape).astype(tensor.data.dtype)
    tensor.data = arr
    return tensor


def _rand_dtype(tensor):
    return tensor.data.dtype


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor.shape)
    bound = gain * math.sqrt(3.0 / (fan_in + fan_out))
    return uniform_(tensor, -bound, bound)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor.shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(tensor: Tensor, a: float = 0.0,
                     mode: str = "fan_in",
                     nonlinearity: str = "leaky_relu") -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor.shape)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity, a)
    bound = gain * math.sqrt(3.0 / max(fan, 1))
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(tensor: Tensor, a: float = 0.0,
                    mode: str = "fan_in",
                    nonlinearity: str = "leaky_relu") -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor.shape)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(max(fan, 1))
    return normal_(tensor, 0.0, std)


def orthogonal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill the tensor with an (approximately) orthogonal matrix.

    For 2D tensors a QR decomposition of a random Gaussian matrix is used;
    for higher-order tensors the first two dimensions are orthogonalised.
    """
    shape = tensor.shape
    rows = shape[0]
    cols = shape[1] if len(shape) >= 2 else 1
    flat = np.random.normal(0.0, 1.0, size=(rows, cols)).astype(tensor.data.dtype)
    q, r = np.linalg.qr(flat)
    # Enforce a deterministic sign convention
    q *= np.sign(np.diag(r))
    q = q * gain
    if len(shape) == 1:
        out = q.reshape(shape)
    else:
        out = q.reshape((rows, cols) + tuple(shape[2:]))
    tensor.data = out.astype(tensor.data.dtype)
    return tensor


__all__ = [
    "zeros_", "ones_", "constant_", "eye_", "uniform_", "normal_",
    "xavier_uniform_", "xavier_normal_", "kaiming_uniform_", "kaiming_normal_",
    "orthogonal_", "calculate_gain", "_calculate_fan_in_and_fan_out",
]
