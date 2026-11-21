"""Unified dispatch — auto-select backend (CPU / CUDA) by device or domain."""

from typing import Optional

from ..c_loader import load_library

# ---- CUDA availability --------------------------------------------------
_HAS_CUDA = False
try:
    from ..cuda_kernels import _HAS_CUDA_BACKEND
    _HAS_CUDA = bool(_HAS_CUDA_BACKEND)
except ImportError:
    pass

# ---- C shared-library availability (generic) -----------------------------
_C_LIB, _HAS_C = load_library("neural_core_kernel")
if not _HAS_C:
    _C_LIB, _HAS_C = load_library("_SNEPPX_c")


def get_backend(device: Optional[str] = None):
    """Return the appropriate backend for *device*.

    Parameters
    ----------
    device:
        ``"cpu"``, ``"cuda"`` (or ``"cuda:0"`` etc.), or *None* (→ ``"cpu"``).
    """
    if device is None:
        device = "cpu"
    if device.startswith("cuda") and _HAS_CUDA:
        from .dispatch_cuda import CudaBackend
        return CudaBackend()
    from .dispatch_cpu import CpuBackend
    return CpuBackend()


def get_security_backend():
    """Return the security-domain backend."""
    from .security_dispatch import SecurityBackend
    return SecurityBackend()


def get_algo_backend():
    """Return the algorithm-domain backend."""
    from .algo_dispatch import AlgoBackend
    return AlgoBackend()


def get_kernel_backend():
    """Return the kernel-domain backend."""
    from .kernel_dispatch import KernelBackend
    return KernelBackend()


def get_infra_backend():
    """Return the infrastructure-domain backend."""
    from .infra_dispatch import InfraBackend
    return InfraBackend()


__all__ = [
    "get_backend",
    "get_security_backend",
    "get_algo_backend",
    "get_kernel_backend",
    "get_infra_backend",
    "_HAS_CUDA",
    "_HAS_C",
]
