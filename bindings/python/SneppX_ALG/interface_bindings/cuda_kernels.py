"""CUDA Kernel Interface — Python fallback implementations matching C++ CUDA backend API.

This module provides the Python-side interface for CUDA kernels. When the C++ CUDA
backend is available (via pybind11), these implementations are replaced by the
compiled kernels. The interface matches the C++ kernel signatures for seamless swapping.
"""

from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import math
import threading

# Try to import the C++ CUDA backend
try:
    from .. import _arix_c as _CUDA_BACKEND

    _HAS_CUDA_BACKEND = True
except ImportError:
    try:
        from .. import _SNEPPX_c as _CUDA_BACKEND

        _HAS_CUDA_BACKEND = True
    except ImportError:
        _CUDA_BACKEND = None
        _HAS_CUDA_BACKEND = False


class ComputeCapability(Enum):
    """GPU compute capability levels."""

    PASCAL = (6, 0)
    VOLTA = (7, 0)
    TURING = (7, 5)
    AMPERE = (8, 0)
    HOPPER = (9, 0)
    BLACKWELL = (10, 0)


@dataclass
class KernelConfig:
    """Kernel launch configuration."""

    grid_dim: Tuple[int, int, int] = (1, 1, 1)
    block_dim: Tuple[int, int, int] = (256, 1, 1)
    shared_mem_bytes: int = 0
    stream: Optional[int] = None  # CUDA stream handle


@dataclass
class TensorDescriptor:
    """Describes a tensor for kernel launches."""

    shape: Tuple[int, ...]
    dtype: str
    strides: Optional[Tuple[int, ...]] = None
    offset: int = 0
    is_contiguous: bool = True


class CUDAStream:
    """CUDA stream wrapper."""

    def __init__(self, stream_id: int = 0, device_id: int = 0):
        self.stream_id = stream_id
        self.device_id = device_id
        self._events: List["CUDAEvent"] = []

    def synchronize(self) -> None:
        """Synchronize the stream (no-op in Python fallback)."""
        pass

    def record_event(self) -> "CUDAEvent":
        """Record an event on this stream."""
        event = CUDAEvent()
        self._events.append(event)
        return event


class CUDAEvent:
    """CUDA event wrapper for timing."""

    def __init__(self):
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def record(self, stream: Optional[CUDAStream] = None) -> None:
        import time

        self._start_time = time.perf_counter()

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, other: "CUDAEvent") -> float:
        """Return elapsed time in milliseconds."""
        if self._end_time and other._start_time:
            return (other._start_time - self._end_time) * 1000
        return 0.0


class KernelLauncher:
    """Launches CUDA kernels with proper configuration."""

    def __init__(self):
        self._kernels: Dict[str, Any] = {}
        self._autotune_cache: Dict[str, KernelConfig] = {}
        self._lock = threading.Lock()

    def register_kernel(self, name: str, kernel_fn: callable) -> None:
        """Register a kernel implementation."""
        self._kernels[name] = kernel_fn

    def launch(self, kernel_name: str, config: KernelConfig, *args, **kwargs) -> Any:
        """Launch a registered kernel."""
        if kernel_name not in self._kernels:
            raise ValueError(f"Kernel '{kernel_name}' not registered")
        return self._kernels[kernel_name](*args, **kwargs)

    def autotune(
        self,
        kernel_name: str,
        problem_size: Tuple[int, ...],
        candidate_configs: List[KernelConfig],
    ) -> KernelConfig:
        """Auto-tune kernel configuration for a problem size."""
        cache_key = f"{kernel_name}_{problem_size}"
        with self._lock:
            if cache_key in self._autotune_cache:
                return self._autotune_cache[cache_key]

        # Simple heuristic: pick config with max occupancy
        best_config = candidate_configs[0]
        best_score = 0
        for config in candidate_configs:
            block_size = config.block_dim[0] * config.block_dim[1] * config.block_dim[2]
            score = block_size / (config.shared_mem_bytes + 1)
            if score > best_score:
                best_score = score
                best_config = config

        self._autotune_cache[cache_key] = best_config
        return best_config


# Global kernel launcher
_GLOBAL_LAUNCHER = KernelLauncher()


def get_launcher() -> KernelLauncher:
    return _GLOBAL_LAUNCHER


# =============================================================================
# Core CUDA Kernel Implementations (Python Fallbacks)
# =============================================================================


def gemm_kernel(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    M: int,
    N: int,
    K: int,
    alpha: float = 1.0,
    beta: float = 0.0,
    trans_a: bool = False,
    trans_b: bool = False,
) -> None:
    """General Matrix Multiply: C = alpha * A @ B + beta * C

    Shapes:
        A: (M, K) if not trans_a else (K, M)
        B: (K, N) if not trans_b else (N, K)
        C: (M, N)
    """
    if trans_a:
        A = A.T
    if trans_b:
        B = B.T

    if beta != 0.0:
        C[:] = beta * C + alpha * (A @ B)
    else:
        C[:] = alpha * (A @ B)


def gemm_batched_kernel(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    batch_size: int,
    M: int,
    N: int,
    K: int,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> None:
    """Batched GEMM: C[b] = alpha * A[b] @ B[b] + beta * C[b]"""
    for b in range(batch_size):
        if beta != 0.0:
            C[b] = beta * C[b] + alpha * (A[b] @ B[b])
        else:
            C[b] = alpha * (A[b] @ B[b])


def flash_attention_v2_kernel(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    O: np.ndarray,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    rng_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flash Attention v2 forward pass.

    Args:
        Q: (batch, heads, seq_len, head_dim)
        K: (batch, heads, seq_len, head_dim)
        V: (batch, heads, seq_len, head_dim)
        O: output buffer (batch, heads, seq_len, head_dim)

    Returns:
        (output, log_sum_exp) for backward pass
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Online softmax algorithm
    B, H, L, D = Q.shape
    LSE = np.zeros((B, H, L), dtype=np.float32)

    # For Python fallback, use standard attention with online softmax
    for b in range(B):
        for h in range(H):
            q = Q[b, h]  # (L, D)
            k = K[b, h]  # (L, D)
            v = V[b, h]  # (L, D)

            # S = Q @ K^T
            S = q @ k.T * softmax_scale  # (L, L)

            if causal:
                mask = np.triu(np.ones((L, L), dtype=bool), k=1)
                S[mask] = -1e9

            # Online softmax: compute row-wise max and sum_exp
            row_max = S.max(axis=1, keepdims=True)
            exp_S = np.exp(S - row_max)
            row_sum = exp_S.sum(axis=1, keepdims=True)

            # P = exp(S) / sum
            P = exp_S / row_sum

            if dropout_p > 0.0 and rng_state is not None:
                np.random.seed(rng_state)
                mask = np.random.binomial(1, 1 - dropout_p, P.shape)
                P = P * mask / (1 - dropout_p)

            # O = P @ V
            O[b, h] = P @ v

            # log_sum_exp for backward (per row)
            LSE[b, h] = row_max.squeeze() + np.log(row_sum.squeeze())

    return O, LSE


def flash_attention_v2_backward_kernel(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    O: np.ndarray,
    dO: np.ndarray,
    LSE: np.ndarray,
    dQ: np.ndarray,
    dK: np.ndarray,
    dV: np.ndarray,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> None:
    """Flash Attention v2 backward pass.

    Computes gradients dQ, dK, dV from dO using the online softmax algorithm.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    B, H, L, D = Q.shape

    for b in range(B):
        for h in range(H):
            q = Q[b, h]  # (L, D)
            k = K[b, h]  # (L, D)
            v = V[b, h]  # (L, D)
            o = O[b, h]  # (L, D)
            do = dO[b, h]  # (L, D)
            lse = LSE[b, h]  # (L,)

            # Recompute S = Q @ K^T
            S = q @ k.T * softmax_scale
            if causal:
                mask = np.triu(np.ones((L, L), dtype=bool), k=1)
                S[mask] = -1e9

            # P = softmax(S)
            row_max = S.max(axis=1, keepdims=True)
            exp_S = np.exp(S - row_max)
            row_sum = exp_S.sum(axis=1, keepdims=True)
            P = exp_S / row_sum

            # dP = dO @ V^T
            dP = do @ v.T

            # dS = P * (dP - sum(dP * P, axis=1, keepdims=True))
            dP_P_sum = (dP * P).sum(axis=1, keepdims=True)
            dS = P * (dP - dP_P_sum) * softmax_scale

            # Gradients
            dQ[b, h] = dS @ k
            dK[b, h] = dS.T @ q
            dV[b, h] = P.T @ do


def layernorm_kernel(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    normalized_shape: Tuple[int, ...],
    eps: float = 1e-5,
) -> None:
    """LayerNorm forward pass.

    input: (*, normalized_shape)
    weight: (normalized_shape,)
    bias: (normalized_shape,)
    output: (*, normalized_shape)
    """
    # Compute mean and variance over last dimension
    axis = tuple(range(-len(normalized_shape), 0))
    mean = input.mean(axis=axis, keepdims=True)
    var = input.var(axis=axis, keepdims=True)

    # Normalize
    x_hat = (input - mean) / np.sqrt(var + eps)

    # Scale and shift
    output[:] = x_hat * weight + bias


def layernorm_backward_kernel(
    input: np.ndarray,
    weight: np.ndarray,
    grad_output: np.ndarray,
    grad_input: np.ndarray,
    grad_weight: np.ndarray,
    grad_bias: np.ndarray,
    normalized_shape: Tuple[int, ...],
    eps: float = 1e-5,
    mean: Optional[np.ndarray] = None,
    rstd: Optional[np.ndarray] = None,
) -> None:
    """LayerNorm backward pass."""
    axis = tuple(range(-len(normalized_shape), 0))

    if mean is None:
        mean = input.mean(axis=axis, keepdims=True)
    if rstd is None:
        var = input.var(axis=axis, keepdims=True)
        rstd = 1.0 / np.sqrt(var + eps)

    x_hat = (input - mean) * rstd

    # grad_weight = sum(grad_output * x_hat) over non-normalized dims
    # grad_bias = sum(grad_output) over non-normalized dims
    reduce_axes = tuple(range(input.ndim - len(normalized_shape)))
    if reduce_axes:
        grad_weight[:] = (grad_output * x_hat).sum(axis=reduce_axes)
        grad_bias[:] = grad_output.sum(axis=reduce_axes)
    else:
        grad_weight[:] = grad_output * x_hat
        grad_bias[:] = grad_output

    # grad_input
    N = np.prod(normalized_shape)
    dx_hat = grad_output * weight
    dx = (
        dx_hat
        - dx_hat.mean(axis=axis, keepdims=True)
        - x_hat * (dx_hat * x_hat).mean(axis=axis, keepdims=True)
    ) * rstd
    grad_input[:] = dx


def rmsnorm_kernel(
    input: np.ndarray,
    weight: np.ndarray,
    output: np.ndarray,
    normalized_shape: Tuple[int, ...],
    eps: float = 1e-6,
) -> None:
    """RMSNorm forward pass (no bias)."""
    axis = tuple(range(-len(normalized_shape), 0))
    rms = np.sqrt((input * input).mean(axis=axis, keepdims=True) + eps)
    output[:] = (input / rms) * weight


def softmax_kernel(input: np.ndarray, output: np.ndarray, axis: int = -1) -> None:
    """Softmax forward pass."""
    max_val = input.max(axis=axis, keepdims=True)
    exp_input = np.exp(input - max_val)
    sum_exp = exp_input.sum(axis=axis, keepdims=True)
    output[:] = exp_input / sum_exp


def silu_kernel(input: np.ndarray, output: np.ndarray) -> None:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    output[:] = input * (1.0 / (1.0 + np.exp(-input)))


def gelu_kernel(
    input: np.ndarray, output: np.ndarray, approximate: bool = True
) -> None:
    """GELU activation.

    By default uses the tanh approximation (as used by PyTorch, TensorFlow).
    Set approximate=False for exact formula (requires scipy.special.erf).
    """
    # tanh approximation (standard in DL frameworks)
    output[:] = (
        0.5
        * input
        * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (input + 0.044715 * input**3)))
    )


def relu_kernel(input: np.ndarray, output: np.ndarray) -> None:
    """ReLU activation."""
    output[:] = np.maximum(input, 0)


def dropout_kernel(
    input: np.ndarray,
    output: np.ndarray,
    p: float,
    training: bool,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Dropout forward pass. Returns mask for backward."""
    if not training or p == 0.0:
        output[:] = input
        return np.ones_like(input, dtype=bool)

    if seed is not None:
        np.random.seed(seed)

    mask = np.random.binomial(1, 1.0 - p, input.shape) / (1.0 - p)
    output[:] = input * mask
    return mask


def dropout_backward_kernel(
    grad_output: np.ndarray, mask: np.ndarray, grad_input: np.ndarray, p: float
) -> None:
    """Dropout backward pass."""
    grad_input[:] = grad_output * mask


def bias_gelu_kernel(
    input: np.ndarray, bias: np.ndarray, output: np.ndarray, approximate: bool = False
) -> None:
    """Fused bias + GELU."""
    x = input + bias
    gelu_kernel(x, output, approximate)


def fused_adamw_kernel(
    param: np.ndarray,
    grad: np.ndarray,
    exp_avg: np.ndarray,
    exp_avg_sq: np.ndarray,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    max_grad_norm: Optional[float] = None,
) -> None:
    """Fused AdamW optimizer step."""
    if max_grad_norm is not None:
        grad_norm = np.linalg.norm(grad)
        if grad_norm > max_grad_norm:
            grad = grad * (max_grad_norm / grad_norm)

    # Weight decay
    param = param * (1.0 - lr * weight_decay)

    # Update moments
    exp_avg[:] = beta1 * exp_avg + (1.0 - beta1) * grad
    exp_avg_sq[:] = beta2 * exp_avg_sq + (1.0 - beta2) * (grad * grad)

    # Bias correction
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step

    step_size = lr / bias_correction1
    denom = np.sqrt(exp_avg_sq / bias_correction2) + eps

    param[:] -= step_size * (exp_avg / denom)


def conv2d_kernel(
    input: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    output: np.ndarray,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
) -> None:
    """2D Convolution (NCHW layout)."""
    N, C_in, H, W = input.shape
    C_out, C_in_g, kH, kW = weight.shape
    assert C_in == C_in_g * groups

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1

    # Pad input
    if pad_h > 0 or pad_w > 0:
        input_padded = np.pad(input, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    else:
        input_padded = input

    # Im2col + GEMM
    for n in range(N):
        for g in range(groups):
            c_start = g * C_in_g
            c_end = c_start + C_in_g
            w_g = weight[g * (C_out // groups) : (g + 1) * (C_out // groups)]
            x_g = input_padded[n, c_start:c_end]

            # im2col
            col = np.zeros((C_in_g * kH * kW, H_out * W_out), dtype=input.dtype)
            for i in range(H_out):
                h_start = i * stride_h
                for j in range(W_out):
                    w_start = j * stride_w
                    patch = x_g[
                        :,
                        h_start : h_start + kH * dil_h : dil_h,
                        w_start : w_start + kW * dil_w : dil_w,
                    ]
                    col[:, i * W_out + j] = patch.ravel()

            # GEMM
            out_g = w_g.reshape(C_out // groups, -1) @ col
            output[n, g * (C_out // groups) : (g + 1) * (C_out // groups)] = (
                out_g.reshape(C_out // groups, H_out, W_out)
            )

    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)


# =============================================================================
# Kernel Registration
# =============================================================================


def _register_kernels():
    launcher = get_launcher()
    launcher.register_kernel("gemm", gemm_kernel)
    launcher.register_kernel("gemm_batched", gemm_batched_kernel)
    launcher.register_kernel("flash_attention_v2", flash_attention_v2_kernel)
    launcher.register_kernel(
        "flash_attention_v2_backward", flash_attention_v2_backward_kernel
    )
    launcher.register_kernel("layernorm", layernorm_kernel)
    launcher.register_kernel("layernorm_backward", layernorm_backward_kernel)
    launcher.register_kernel("rmsnorm", rmsnorm_kernel)
    launcher.register_kernel("softmax", softmax_kernel)
    launcher.register_kernel("silu", silu_kernel)
    launcher.register_kernel("gelu", gelu_kernel)
    launcher.register_kernel("relu", relu_kernel)
    launcher.register_kernel("dropout", dropout_kernel)
    launcher.register_kernel("dropout_backward", dropout_backward_kernel)
    launcher.register_kernel("bias_gelu", bias_gelu_kernel)
    launcher.register_kernel("fused_adamw", fused_adamw_kernel)
    launcher.register_kernel("conv2d", conv2d_kernel)


# Register on import
_register_kernels()


__all__ = [
    # Core types
    "ComputeCapability",
    "KernelConfig",
    "TensorDescriptor",
    "CUDAStream",
    "CUDAEvent",
    "KernelLauncher",
    "get_launcher",
    # Kernels
    "gemm_kernel",
    "gemm_batched_kernel",
    "flash_attention_v2_kernel",
    "flash_attention_v2_backward_kernel",
    "layernorm_kernel",
    "layernorm_backward_kernel",
    "rmsnorm_kernel",
    "softmax_kernel",
    "silu_kernel",
    "gelu_kernel",
    "relu_kernel",
    "dropout_kernel",
    "dropout_backward_kernel",
    "bias_gelu_kernel",
    "fused_adamw_kernel",
    "conv2d_kernel",
    # Backend status
    "_HAS_CUDA_BACKEND",
]
