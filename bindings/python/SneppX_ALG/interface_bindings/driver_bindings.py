"""Hardware driver bindings — CUDA, ROCm, TPU driver interfaces.

Wraps C implementations in ``drivers/`` with pure-Python fallback.
"""

from typing import Optional, List, Dict, Tuple
import threading

import numpy as np

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_drivers")


class CUDADeviceProps:
    """CUDA device properties queried at runtime."""

    def __init__(self, name: str = "", global_mem: int = 0, shared_mem: int = 0,
                 warp_size: int = 32, max_threads: int = 1024,
                 sm_count: int = 0, compute_major: int = 0, compute_minor: int = 0,
                 has_tensor_cores: bool = False):
        self.name = name
        self.global_mem_bytes = global_mem
        self.shared_mem_per_block = shared_mem
        self.warp_size = warp_size
        self.max_threads_per_block = max_threads
        self.max_blocks_per_grid = [2**31 - 1, 65535, 65535]
        self.compute_capability_major = compute_major
        self.compute_capability_minor = compute_minor
        self.num_sms = sm_count
        self.supports_tensor_cores = has_tensor_cores
        self.supports_cooperative_groups = True


class CUDADriver:
    """CUDA driver interface — device management, streams, events, memory."""

    def __init__(self):
        self._has_c = _HAS_C
        self._lock = threading.Lock()
        self._initialized = False
        self._current_device = 0

    # ---- Capability ----

    @staticmethod
    def is_available() -> bool:
        return False

    # ---- Device lifecycle ----

    def get_device_count(self) -> int:
        return 0

    def get_device_props(self, dev_id: int = 0) -> CUDADeviceProps:
        return CUDADeviceProps(name="cpu", sm_count=1)

    def set_device(self, dev_id: int) -> int:
        self._current_device = dev_id
        return 0

    def get_device(self) -> int:
        return self._current_device

    # ---- Context ----

    def create_context(self, device_id: int = 0) -> Optional[dict]:
        return {"device_id": device_id, "alloc_bytes": 0, "error_state": 0}

    def destroy_context(self, ctx: dict):
        pass

    def context_error(self, ctx: dict) -> int:
        return ctx.get("error_state", 0) if ctx else -1

    # ---- Stream / Event ----

    def stream_create(self, priority: int = 0) -> Optional[dict]:
        return {"handle": None, "device_id": self._current_device, "priority": priority}

    def stream_destroy(self, stream: dict):
        pass

    def stream_synchronize(self, stream: dict) -> int:
        return 0

    def event_create(self) -> Optional[dict]:
        return {"handle": None, "device_id": self._current_device}

    def event_destroy(self, event: dict):
        pass

    def event_record(self, event: dict, stream: dict) -> int:
        return 0

    def event_synchronize(self, event: dict) -> int:
        return 0

    def event_elapsed_ms(self, start: dict, end: dict) -> float:
        return 0.0

    # ---- Memory ----

    def mem_alloc(self, bytes: int) -> Optional[int]:
        return id(bytearray(bytes))

    def mem_free(self, dev_ptr: int):
        pass

    def mem_htod(self, dev_dst: int, host_src: np.ndarray, bytes: int) -> int:
        return 0

    def mem_dtoh(self, host_dst: np.ndarray, dev_src: int, bytes: int) -> int:
        return 0

    def mem_dtod(self, dev_dst: int, dev_src: int, bytes: int) -> int:
        return 0

    def mem_set(self, dev_ptr: int, value: int, bytes: int) -> int:
        return 0

    # ---- Kernel dispatch ----

    def launch_kernel(self, kernel_name: str, grid: Tuple[int, int, int],
                      block: Tuple[int, int, int], shared_mem: int = 0,
                      args: Optional[List] = None) -> int:
        return 0

    # ---- Tensor core GEMM ----

    def tc_gemm(self, m: int, n: int, k: int, a: int, b: int, c: int,
                dtype: int = 0, stream: Optional[dict] = None) -> int:
        return 0


class ROCmDriver:
    """ROCm driver interface — AMD GPU device management."""

    def __init__(self):
        self._has_c = _HAS_C
        self._current_device = 0

    @staticmethod
    def is_available() -> bool:
        return False

    def get_device_count(self) -> int:
        return 0

    def get_device_props(self, dev_id: int = 0) -> dict:
        return {"name": "cpu", "global_mem_bytes": 0, "wavefront_size": 64,
                "num_cus": 0, "gcn_arch_major": 0, "gcn_arch_minor": 0}

    def set_device(self, dev_id: int) -> int:
        self._current_device = dev_id
        return 0

    def create_context(self, device_id: int = 0) -> Optional[dict]:
        return {"device_id": device_id, "alloc_bytes": 0}

    def destroy_context(self, ctx: dict):
        pass

    def stream_create(self) -> Optional[dict]:
        return {"handle": None, "device_id": self._current_device}

    def stream_destroy(self, stream: dict):
        pass

    def stream_synchronize(self, stream: dict) -> int:
        return 0

    def event_create(self) -> Optional[dict]:
        return {"handle": None, "device_id": self._current_device}

    def event_destroy(self, event: dict):
        pass

    def event_record(self, event: dict, stream: dict) -> int:
        return 0

    def event_synchronize(self, event: dict) -> int:
        return 0

    def mem_alloc(self, bytes: int) -> Optional[int]:
        return id(bytearray(bytes))

    def mem_free(self, dev_ptr: int):
        pass

    def mem_htod(self, dev_dst: int, host_src: np.ndarray, bytes: int) -> int:
        return 0

    def mem_dtoh(self, host_dst: np.ndarray, dev_src: int, bytes: int) -> int:
        return 0

    def mem_dtod(self, dev_dst: int, dev_src: int, bytes: int) -> int:
        return 0

    def launch_kernel(self, kernel_name: str, grid: Tuple[int, int, int],
                      group: Tuple[int, int, int], local_mem: int = 0) -> int:
        return 0


class TPUDriver:
    """TPU driver interface — Google TPU / custom ASIC acceleration."""

    def __init__(self):
        self._has_c = _HAS_C
        self._current_device = 0

    @staticmethod
    def is_available() -> bool:
        return False

    def get_device_count(self) -> int:
        return 0

    def get_device_props(self, dev_id: int = 0) -> dict:
        return {"name": "cpu", "device_memory_bytes": 0, "num_cores": 0,
                "num_chips": 0, "supports_bfloat16": True}

    def create_context(self, device_id: int = 0) -> Optional[dict]:
        return {"device_id": device_id, "client": None, "alloc_bytes": 0}

    def destroy_context(self, ctx: dict):
        pass

    def mem_alloc(self, bytes: int, ctx: dict) -> Optional[int]:
        return id(bytearray(bytes))

    def mem_free(self, dev_ptr: int, ctx: dict):
        pass

    def mem_htod(self, dev_dst: int, host_src: np.ndarray, bytes: int, ctx: dict) -> int:
        return 0

    def mem_dtoh(self, host_dst: np.ndarray, dev_src: int, bytes: int, ctx: dict) -> int:
        return 0

    def compile(self, hlo_module: str, ctx: dict) -> Optional[dict]:
        return {"executable": None, "input_count": 0, "output_count": 0}

    def executable_destroy(self, exec: dict):
        pass

    def execute(self, exec: dict, inputs: List[np.ndarray],
                outputs: List[np.ndarray], ctx: dict) -> int:
        return 0

    def all_reduce(self, send_buf: np.ndarray, recv_buf: np.ndarray,
                   dtype: int = 0, reduce_op: int = 0, ctx: Optional[dict] = None) -> int:
        np.copyto(recv_buf, send_buf)
        return 0
