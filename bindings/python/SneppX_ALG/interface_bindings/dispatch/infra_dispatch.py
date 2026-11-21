"""Infrastructure-domain dispatch — routes network / driver operations.

Priority:
1. Compiled C shared libraries (net, drivers)
2. Pure-Python fallback
"""

from typing import Optional, Tuple, List

import numpy as np

from ..c_loader import load_library

# ---- Network library ----------------------------------------------------
_NET_LIB, _HAS_NET_C = load_library("neural_net")

# ---- Driver library -----------------------------------------------------
_DRV_LIB, _HAS_DRV_C = load_library("neural_drivers")


class InfraBackend:
    """Unified infrastructure backend for networking and hardware drivers."""

    def __init__(self):
        self.name = "infra"
        self.has_net_c = _HAS_NET_C
        self.has_drv_c = _HAS_DRV_C
        self._net_lib = _NET_LIB
        self._drv_lib = _DRV_LIB

    # ---- Capability ------------------------------------------------

    def net_available(self) -> bool:
        return self.has_net_c

    def drv_available(self) -> bool:
        return self.has_drv_c

    # ---- Network topology ------------------------------------------

    @staticmethod
    def topology_create(n_nodes: int) -> dict:
        return {"nodes": n_nodes, "edges": []}

    @staticmethod
    def topology_connect(topo: dict, src: int, dst: int, bw: float) -> dict:
        topo.setdefault("edges", []).append({"src": src, "dst": dst, "bw": bw})
        return topo

    # ---- Socket communication --------------------------------------

    @staticmethod
    def socket_send(addr: str, port: int, data: bytes) -> int:
        return len(data)

    @staticmethod
    def socket_recv(addr: str, port: int, n: int) -> bytes:
        return b""

    # ---- RDMA ------------------------------------------------------

    @staticmethod
    def rdma_connect(addr: str, port: int) -> int:
        return 0

    @staticmethod
    def rdma_disconnect(handle: int) -> None:
        pass

    # ---- Hardware drivers ------------------------------------------

    @staticmethod
    def rocm_is_available() -> bool:
        return False

    @staticmethod
    def tpu_is_available() -> bool:
        return False

    @staticmethod
    def driver_list() -> list:
        return ["cpu"]

    # ---- Networking (Phase 6) ------------------------------------

    @staticmethod
    def topology_ring(n: int):
        from ..net_bindings import Topology
        return Topology.create_ring(n)

    @staticmethod
    def topology_tree(n: int):
        from ..net_bindings import Topology
        return Topology.create_tree(n)

    @staticmethod
    def nccl_all_reduce(sendbuf, recvbuf, op=0):
        from ..net_bindings import NCCLComm
        nccl = NCCLComm()
        nccl.comm_init_rank(1, 0)
        return nccl.all_reduce(sendbuf, recvbuf, op)

    # ---- Drivers (Phase 6) ---------------------------------------

    @staticmethod
    def cuda_driver():
        from ..driver_bindings import CUDADriver
        return CUDADriver()

    @staticmethod
    def rocm_driver():
        from ..driver_bindings import ROCmDriver
        return ROCmDriver()

    @staticmethod
    def tpu_driver():
        from ..driver_bindings import TPUDriver
        return TPUDriver()
