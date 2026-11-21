"""Quantization Module — INT8/FP8/AWQ/GPTQ quantization and compression.

Provides both pure NumPy implementations and wrappers around C/CUDA kernels
when the C backend is available.
"""

from typing import Optional, Tuple, List, Union
from .tensor import Tensor, _HAS_C_BACKEND
import numpy as np
import math
import ctypes


class QuantMode:
    NONE = 0
    INT8_SYM = 1
    INT8_ASYM = 2
    INT4_SYM = 3
    FP8_E4M3 = 4
    FP8_E5M2 = 5
    AWQ = 6
    GPTQ = 7


class QuantGranularity:
    PER_TENSOR = 0
    PER_CHANNEL = 1
    PER_GROUP = 2
    PER_TOKEN = 3


# ============================================================================
# INT8 Quantization (pure NumPy)
# ============================================================================


def quantize_int8_sym(tensor: Tensor) -> Tuple[Tensor, float]:
    """Symmetric INT8 quantization: x_q = round(x / scale), scale = max(|x|) / 127."""
    arr = tensor.data
    max_abs = np.max(np.abs(arr))
    if max_abs < 1e-10:
        return Tensor(np.zeros_like(arr, dtype=np.int8)), 1.0
    scale = max_abs / 127.0
    q_arr = np.round(arr / scale).clip(-128, 127).astype(np.int8)
    return Tensor.from_numpy(q_arr, dtype="int8"), scale


def dequantize_int8_sym(qtensor: Tensor, scale: float) -> Tensor:
    """Dequantize symmetric INT8: x = x_q * scale."""
    arr = qtensor.data.astype(np.float32) * scale
    return Tensor.from_numpy(arr)


def quantize_int8_asym(tensor: Tensor) -> Tuple[Tensor, float, int]:
    """Asymmetric INT8 quantization with zero-point."""
    arr = tensor.data
    vmin, vmax = np.min(arr), np.max(arr)
    if vmax - vmin < 1e-10:
        return Tensor(np.zeros_like(arr, dtype=np.int8)), 1.0, 0
    scale = (vmax - vmin) / 255.0
    zp = int(np.round(-vmin / scale))
    zp = max(0, min(255, zp))
    q_arr = np.round(arr / scale + zp).clip(0, 255).astype(np.uint8)
    return Tensor.from_numpy(q_arr, dtype="uint8"), scale, zp


def dequantize_int8_asym(qtensor: Tensor, scale: float, zp: int) -> Tensor:
    """Dequantize asymmetric INT8: x = (x_q - zp) * scale."""
    arr = qtensor.data.astype(np.float32)
    result = (arr - zp) * scale
    return Tensor.from_numpy(result)


def quantize_int8_channel(tensor: Tensor, dim: int = -1) -> Tuple[Tensor, Tensor]:
    """Per-channel INT8 symmetric quantization along dimension dim."""
    arr = tensor.data
    if dim < 0:
        dim = arr.ndim + dim
    max_abs = np.max(
        np.abs(arr), axis=tuple(d for d in range(arr.ndim) if d != dim), keepdims=True
    )
    scales = np.where(max_abs < 1e-10, 1.0, max_abs / 127.0)
    q_arr = np.round(arr / scales).clip(-128, 127).astype(np.int8)
    return Tensor.from_numpy(q_arr, dtype="int8"), Tensor.from_numpy(scales.squeeze())


# ============================================================================
# INT4 Quantization (packed, symmetric)
# ============================================================================


def quantize_int4_sym(tensor: Tensor) -> Tuple[Tensor, float]:
    """Symmetric INT4 quantization with 2 values packed per byte."""
    arr = tensor.data.flatten()
    max_abs = np.max(np.abs(arr))
    if max_abs < 1e-10:
        n_packed = (len(arr) + 1) // 2
        return Tensor(np.zeros(n_packed, dtype=np.uint8)), 1.0
    scale = max_abs / 7.0
    q = np.round(arr / scale).clip(-8, 7).astype(np.int8) & 0x0F
    packed = (q[1::2].astype(np.uint8) << 4) | q[0::2].astype(np.uint8)
    return Tensor.from_numpy(packed, dtype="uint8"), scale


def dequantize_int4_sym(qtensor: Tensor, scale: float, n: int) -> Tensor:
    """Dequantize packed INT4 back to float32."""
    packed = qtensor.data.astype(np.uint8)
    q0 = (packed & 0x0F).astype(np.int8)
    q1 = ((packed >> 4) & 0x0F).astype(np.int8)
    q0 = np.where(q0 >= 8, q0 - 16, q0)
    q1 = np.where(q1 >= 8, q1 - 16, q1)
    interleaved = np.empty(n, dtype=np.int8)
    interleaved[0::2] = q0[: (n + 1) // 2]
    interleaved[1::2] = q1[: n // 2]
    return Tensor.from_numpy(interleaved.astype(np.float32) * scale)


# ============================================================================
# FP8 Quantization (E4M3 / E5M2)
# ============================================================================


def _float_to_fp8_e4m3(value: float) -> int:
    """Encode a single float32 to FP8 E4M3."""
    b = np.float32(value).view(np.uint32)
    sign = (b >> 31) & 1
    exp_bits = (b >> 23) & 0xFF
    if exp_bits == 0xFF:
        return int((sign << 7) | 0x78)  # Inf
    if exp_bits == 0:
        return int(sign << 7)  # Zero/subnormal
    exp = int(exp_bits) - 127
    mant = (b >> 20) & 0x07
    if exp < -6:
        return int(sign << 7)
    if exp > 8:
        return int((sign << 7) | 0x78)  # Inf
    e4m3_exp = exp + 7
    return int((sign << 7) | (e4m3_exp << 3) | mant)


def _fp8_e4m3_to_float(fp8: int) -> float:
    """Decode a single FP8 E4M3 to float32."""
    sign = (fp8 >> 7) & 1
    e4m3_exp = (fp8 >> 3) & 0x0F
    e4m3_mant = fp8 & 0x07
    if e4m3_exp == 0x0F:
        if e4m3_mant == 0:
            return float("inf") if sign == 0 else float("-inf")
        return float("nan")
    if e4m3_exp == 0 and e4m3_mant == 0:
        return 0.0
    exp = e4m3_exp - 7
    f32_exp = exp + 127
    if f32_exp <= 0:
        return 0.0
    if f32_exp >= 255:
        return float("inf") if sign == 0 else float("-inf")
    f32 = int((sign << 31) | (f32_exp << 23) | (e4m3_mant << 20))
    return np.uint32(f32).view(np.float32).item()


def _float_to_fp8_e5m2(value: float) -> int:
    """Encode a single float32 to FP8 E5M2."""
    b = np.float32(value).view(np.uint32)
    sign = (b >> 31) & 1
    exp_bits = (b >> 23) & 0xFF
    if exp_bits == 0xFF:
        return int((sign << 7) | 0x7C)  # Inf
    if exp_bits == 0:
        return int(sign << 7)
    exp = int(exp_bits) - 127
    mant = (b >> 22) & 0x01
    if exp < -14:
        return int(sign << 7)
    if exp > 15:
        return int((sign << 7) | 0x7C)  # Inf
    e5m2_exp = exp + 15
    return int((sign << 7) | (e5m2_exp << 2) | mant)


def _fp8_e5m2_to_float(fp8: int) -> float:
    """Decode a single FP8 E5M2 to float32."""
    sign = (fp8 >> 7) & 1
    e5m2_exp = (fp8 >> 2) & 0x1F
    e5m2_mant = fp8 & 0x03
    if e5m2_exp == 0x1F:
        if e5m2_mant == 0:
            return float("inf") if sign == 0 else float("-inf")
        return float("nan")
    if e5m2_exp == 0 and e5m2_mant == 0:
        return 0.0
    exp = e5m2_exp - 15
    f32_exp = exp + 127
    if f32_exp <= 0:
        return 0.0
    if f32_exp >= 255:
        return float("inf") if sign == 0 else float("-float")
    f32 = int((sign << 31) | (f32_exp << 23) | (e5m2_mant << 21))
    return np.uint32(f32).view(np.float32).item()


def quantize_fp8_e4m3(tensor: Tensor) -> Tensor:
    """Quantize tensor to FP8 E4M3 format."""
    vec = tensor.data.flatten()
    result = np.array([_float_to_fp8_e4m3(v) for v in vec], dtype=np.uint8)
    return Tensor.from_numpy(result.reshape(tensor.shape), dtype="uint8")


def dequantize_fp8_e4m3(qtensor: Tensor) -> Tensor:
    """Dequantize from FP8 E4M3 back to float32."""
    vec = qtensor.data.flatten().astype(np.uint32)
    result = np.array([_fp8_e4m3_to_float(v) for v in vec], dtype=np.float32)
    return Tensor.from_numpy(result.reshape(qtensor.shape))


def quantize_fp8_e5m2(tensor: Tensor) -> Tensor:
    """Quantize tensor to FP8 E5M2 format."""
    vec = tensor.data.flatten()
    result = np.array([_float_to_fp8_e5m2(v) for v in vec], dtype=np.uint8)
    return Tensor.from_numpy(result.reshape(tensor.shape), dtype="uint8")


def dequantize_fp8_e5m2(qtensor: Tensor) -> Tensor:
    """Dequantize from FP8 E5M2 back to float32."""
    vec = qtensor.data.flatten().astype(np.uint32)
    result = np.array([_fp8_e5m2_to_float(v) for v in vec], dtype=np.float32)
    return Tensor.from_numpy(result.reshape(qtensor.shape))


# ============================================================================
# AWQ: Activation-aware Weight Quantization
# ============================================================================


def _compute_best_scale(
    w_row: np.ndarray, col: int, group_size: int, act_scale: float
) -> float:
    """Grid search for optimal per-channel scale."""
    cols = len(w_row)
    g_start = (col // group_size) * group_size
    g_end = min(g_start + group_size, cols)
    block = w_row[g_start:g_end]
    org_max = np.max(np.abs(block))
    if org_max < 1e-10:
        return 1.0
    alpha = min(act_scale, 1.0)
    best_s = 1.0
    best_loss = 1e20
    for s in np.linspace(alpha, 1.0, 21):
        q_scale = np.max(np.abs(block * s)) / 127.0
        if q_scale < 1e-10:
            q_scale = 1.0
        qi = np.round(block * s / q_scale).clip(-128, 127)
        dq = qi.astype(np.float32) * q_scale / s
        loss = np.mean((dq - block) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_s = s
    return best_s


def awq_scale_weights(
    weights: Tensor, act_scales: Tensor, group_size: int = 128
) -> Tensor:
    """Scale weights by optimal per-channel factors (AWQ preprocessing)."""
    w = weights.data.copy()
    act = act_scales.data.flatten()
    rows, cols = w.shape
    scale_out = np.ones_like(w)
    for r in range(rows):
        for c in range(cols):
            s = _compute_best_scale(
                w[r], c, group_size, act[c] if c < len(act) else 0.5
            )
            scale_out[r, c] = s
            w[r, c] *= s
    return Tensor.from_numpy(w)


def awq_quantize(
    weights: Tensor, act_scales: Tensor, group_size: int = 128
) -> Tuple[Tensor, Tensor]:
    """AWQ: scale + quantize weights to INT8 per group."""
    w = weights.data
    act = act_scales.data.flatten()
    rows, cols = w.shape
    num_groups = (cols + group_size - 1) // group_size
    qw = np.zeros_like(w, dtype=np.int8)
    scales = np.zeros((rows, num_groups), dtype=np.float32)
    for r in range(rows):
        for g in range(num_groups):
            gs = g * group_size
            ge = min(gs + group_size, cols)
            block = w[r, gs:ge].copy()
            for c in range(gs, ge):
                s = _compute_best_scale(
                    w[r], c, group_size, act[c] if c < len(act) else 0.5
                )
                block[c - gs] *= s
            max_abs = np.max(np.abs(block))
            scale = max_abs / 127.0 if max_abs > 1e-10 else 1.0
            scales[r, g] = scale
            qi = np.round(block / scale).clip(-128, 127).astype(np.int8)
            qw[r, gs:ge] = qi
    return Tensor.from_numpy(qw, dtype="int8"), Tensor.from_numpy(scales)


# ============================================================================
# GPTQ: Post-Training Quantization
# ============================================================================


def _cholesky_inv(h: np.ndarray) -> np.ndarray:
    """Compute inverse of Cholesky factor: H^{-1} = (L^{-T})(L^{-1})."""
    dim = h.shape[0]
    try:
        L = np.linalg.cholesky(h)
        L_inv = np.linalg.inv(L)
        return L_inv.T @ L_inv
    except np.linalg.LinAlgError:
        return np.eye(dim)


def gptq_compute_hessian(activations: Tensor, reg: float = 1e-5) -> Tensor:
    """Compute Hessian matrix H = X^T X + reg*I from activation samples."""
    X = activations.data
    n, dim = X.shape
    H = X.T @ X
    H += reg * np.eye(dim) * np.trace(H) / dim
    return Tensor.from_numpy(H)


def gptq_quantize(
    weights: Tensor,
    hessian: Optional[Tensor] = None,
    group_size: int = 128,
    bits: int = 8,
    sym: bool = True,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """GPTQ: one-shot weight quantization with Hessian-based error compensation.

    Args:
        weights: Weight tensor of shape (rows, cols)
        hessian: Optional pre-computed Hessian. If None, uses identity.
        group_size: Column group size for independent quantization.
        bits: Number of bits (4, 8).
        sym: Symmetric quantization.

    Returns:
        (qweight, scales, zeros) — quantized weights, per-group scales, zero-points.
    """
    w = weights.data.astype(np.float64)
    rows, cols = w.shape
    qmax = 2 ** (bits - 1) - 1 if sym else 2**bits - 1
    qmin = -qmax if sym else 0
    num_groups = (cols + group_size - 1) // group_size
    qw = np.zeros_like(w, dtype=np.float32)
    scales = np.zeros((rows, num_groups), dtype=np.float32)
    zeros = np.zeros((rows, num_groups), dtype=np.int32) if not sym else None

    if hessian is not None:
        H = hessian.data.astype(np.float64)
        if H.shape[0] < cols:
            H_big = np.eye(cols, dtype=np.float64) * 1e-5
            H_big[: H.shape[0], : H.shape[0]] = H
            H = H_big
    else:
        H = np.eye(cols, dtype=np.float64)

    H_inv = _cholesky_inv(H)

    for r in range(rows):
        w_row = w[r].copy()
        for g in range(num_groups):
            gs = g * group_size
            ge = min(gs + group_size, cols)
            block = w_row[gs:ge].copy()
            gs_eff = ge - gs
            H_block = H_inv[gs:ge, gs:ge].copy()

            max_abs = np.max(np.abs(block))
            if max_abs < 1e-10:
                scales[r, g] = 1.0
                if zeros is not None:
                    zeros[r, g] = 0
                continue

            scale = max_abs / qmax
            scales[r, g] = scale

            for j in range(gs_eff):
                qi = int(np.round(block[j] / scale))
                qi = max(qmin, min(qmax, qi))
                q_val = float(qi) * scale
                err = q_val - block[j]
                qw[r, gs + j] = q_val
                if j + 1 < gs_eff and abs(H_block[j, j]) > 1e-10:
                    coef = -err / H_block[j, j]
                    block[j + 1 :] += coef * H_block[j + 1 :, j]

    return (
        Tensor.from_numpy(qw),
        Tensor.from_numpy(scales),
        Tensor.from_numpy(zeros) if zeros is not None else None,
    )


# ============================================================================
# Quantized Linear Layer
# ============================================================================


class QuantizedLinear:
    """INT8 quantized linear layer (simulates W8A16 inference)."""

    def __init__(
        self,
        weight: Tensor,
        scale: Union[float, Tensor],
        bias: Optional[Tensor] = None,
        mode: int = QuantMode.INT8_SYM,
    ):
        self.weight = weight
        self.scale = scale
        self.bias = bias
        self.mode = mode

    @classmethod
    def from_float(cls, linear_layer, mode: int = QuantMode.INT8_SYM):
        """Quantize a Linear layer's weight."""
        from .nn import Linear

        w = linear_layer.weight.data
        if mode == QuantMode.INT8_SYM:
            qw, scale = quantize_int8_sym(linear_layer.weight)
        elif mode == QuantMode.INT8_ASYM:
            qw, scale, _ = quantize_int8_asym(linear_layer.weight)
        elif mode == QuantMode.INT4_SYM:
            qw, scale = quantize_int4_sym(linear_layer.weight)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return cls(qw, scale, linear_layer.bias, mode)

    def forward(self, x: Tensor) -> Tensor:
        """W8A16: dequantize weights, compute matmul in float32."""
        if isinstance(self.scale, Tensor):
            w = dequantize_int8_channel(self.weight, self.scale)
        else:
            w = dequantize_int8_sym(self.weight, self.scale)
        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


def dequantize_int8_channel(qtensor: Tensor, scales: Tensor) -> Tensor:
    """Per-channel dequantization of INT8 tensor."""
    arr = qtensor.data.astype(np.float32)
    s = scales.data
    while s.ndim < arr.ndim:
        s = np.expand_dims(s, axis=-1)
    return Tensor.from_numpy(arr * s)


# ============================================================================
# Utility
# ============================================================================


def quantize_error(original: Tensor, quantized: Tensor, metric: str = "mse") -> float:
    """Compute quantization error between original and dequantized tensors."""
    orig = original.data.flatten()
    recon = quantized.data.flatten()
    if metric == "mse":
        return float(np.mean((orig - recon) ** 2))
    elif metric == "mae":
        return float(np.mean(np.abs(orig - recon)))
    elif metric == "snr":
        signal = np.mean(orig**2)
        noise = np.mean((orig - recon) ** 2)
        return float(10 * np.log10(signal / noise)) if noise > 0 else float("inf")
    return 0.0


__all__ = [
    "QuantMode",
    "QuantGranularity",
    "quantize_int8_sym",
    "dequantize_int8_sym",
    "quantize_int8_asym",
    "dequantize_int8_asym",
    "quantize_int8_channel",
    "dequantize_int8_channel",
    "quantize_int4_sym",
    "dequantize_int4_sym",
    "quantize_fp8_e4m3",
    "dequantize_fp8_e4m3",
    "quantize_fp8_e5m2",
    "dequantize_fp8_e5m2",
    "awq_scale_weights",
    "awq_quantize",
    "gptq_compute_hessian",
    "gptq_quantize",
    "QuantizedLinear",
    "quantize_error",
]
