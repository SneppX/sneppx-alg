"""CUDA backend — delegates to compiled CUDA kernels or simulated fallback."""

from typing import Optional, Tuple

import numpy as np

from .dispatch_cpu import CpuBackend

_HAS_CUDA_KERNELS = False
_CUDA_DEVICE = None
_CUDA_KERNELS = None

try:
    from ..cuda_device import (
        cuda_is_available,
        set_device as cuda_set_device,
        current_device as cuda_current_device,
        device_count as cuda_device_count,
    )
    from ..cuda_kernels import (
        _HAS_CUDA_BACKEND,
        get_launcher,
        gemm_kernel,
        flash_attention_v2_kernel,
        layernorm_kernel,
        rmsnorm_kernel,
        softmax_kernel,
        dropout_kernel,
    )
    _HAS_CUDA_KERNELS = bool(_HAS_CUDA_BACKEND)
    _CUDA_DEVICE = "cuda_device"
    _CUDA_KERNELS = "cuda_kernels"
except ImportError:
    pass


class CudaBackend(CpuBackend):
    """Backend that runs operations on CUDA devices.

    When the compiled ``_arix_c`` / ``_SNEPPX_c`` CUDA backend is
    available, real GPU kernels are used.  Otherwise a simulated CUDA
    device layer (``cuda_device.py``) provides API compatibility.
    """

    def __init__(self):
        super().__init__()
        self.name = "cuda"
        self.has_cuda_kernels = _HAS_CUDA_KERNELS

    # ---- Device management ----------------------------------------------

    @staticmethod
    def is_available() -> bool:
        return _HAS_CUDA_KERNELS or cuda_is_available()

    @staticmethod
    def device_count() -> int:
        return cuda_device_count()

    @staticmethod
    def set_device(device_id: int) -> None:
        cuda_set_device(device_id)

    @staticmethod
    def current_device() -> int:
        return cuda_current_device()

    # ---- Kernel dispatch ------------------------------------------------

    @staticmethod
    def gemm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if _HAS_CUDA_KERNELS:
            return gemm_kernel(a, b)
        return a @ b

    @staticmethod
    def flash_attention(q, k, v, mask=None):
        if _HAS_CUDA_KERNELS:
            return flash_attention_v2_kernel(q, k, v, mask)
        return softmax_kernel(q @ k.T / (k.shape[-1] ** 0.5)) @ v

    @staticmethod
    def layernorm(x, weight, bias, eps=1e-5):
        if _HAS_CUDA_KERNELS:
            return layernorm_kernel(x, weight, bias, eps)
        return CpuBackend.layernorm(x, weight, bias, eps)

    @staticmethod
    def rmsnorm(x, weight, eps=1e-5):
        if _HAS_CUDA_KERNELS:
            return rmsnorm_kernel(x, weight, eps)
        return CpuBackend.rmsnorm(x, weight, eps)

    @staticmethod
    def dropout(x, p=0.5):
        if _HAS_CUDA_KERNELS:
            return dropout_kernel(x, p)
        mask = np.random.binomial(1, 1 - p, x.shape) / (1 - p)
        return x * mask

    # ---- Override CPU-only methods to do nothing special -----------------
    # (All base-class methods from CpuBackend work fine as NumPy fallbacks)
