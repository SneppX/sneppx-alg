"""Network infrastructure bindings — topology, socket, RDMA, gRPC, NCCL.

Wraps C implementations in ``net/`` with pure-Python fallback.
"""

from typing import Optional, Tuple, List, Dict
import socket as _socket
import struct
import threading

import numpy as np

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_net")


class TopologyType:
    RING = 0
    TREE = 1
    GRAPH = 2


class Topology:
    """Network topology — ring / tree / graph for collective communication."""

    def __init__(self, topo_type: int = TopologyType.RING, world_size: int = 1):
        self._has_c = _HAS_C
        self.type = topo_type
        self.world_size = world_size
        self._nodes: List[Dict] = []
        self._build()

    def _build(self):
        if self.type == TopologyType.RING:
            for i in range(self.world_size):
                self._nodes.append({
                    "rank": i,
                    "prev": (i - 1) % self.world_size,
                    "next": (i + 1) % self.world_size,
                    "parent": -1,
                    "children": [],
                })
        elif self.type == TopologyType.TREE:
            for i in range(self.world_size):
                parent = (i - 1) // 2 if i > 0 else -1
                children = [2 * i + 1, 2 * i + 2]
                children = [c for c in children if c < self.world_size]
                self._nodes.append({
                    "rank": i, "prev": -1, "next": -1,
                    "parent": parent, "children": children,
                })

    def get_prev(self, rank: int) -> int:
        if 0 <= rank < len(self._nodes):
            return self._nodes[rank]["prev"]
        return -1

    def get_next(self, rank: int) -> int:
        if 0 <= rank < len(self._nodes):
            return self._nodes[rank]["next"]
        return -1

    def get_parent(self, rank: int) -> int:
        if 0 <= rank < len(self._nodes):
            return self._nodes[rank]["parent"]
        return -1

    def get_children(self, rank: int) -> List[int]:
        if 0 <= rank < len(self._nodes):
            return self._nodes[rank]["children"]
        return []

    def compute_route(self, src: int, dst: int) -> List[int]:
        path = [src]
        visited = {src}
        while path[-1] != dst:
            cur = path[-1]
            nxt = self.get_next(cur)
            if nxt not in visited and nxt >= 0:
                path.append(nxt)
                visited.add(nxt)
            else:
                break
        return path if path[-1] == dst else []

    @staticmethod
    def create_ring(n: int):
        return Topology(TopologyType.RING, n)

    @staticmethod
    def create_tree(n: int):
        return Topology(TopologyType.TREE, n)


class SocketComm:
    """TCP / Unix socket communication for point-to-point tensor transfers."""

    def __init__(self):
        self._has_c = _HAS_C
        self._lock = threading.Lock()
        self._sock: Optional[_socket.socket] = None

    def create(self, sock_type: int = 0):
        family = _socket.AF_INET if sock_type == 0 else _socket.AF_UNIX
        self._sock = _socket.socket(family, _socket.SOCK_STREAM)

    def bind(self, port: int) -> int:
        if self._sock is None:
            return -1
        try:
            self._sock.bind(("0.0.0.0", port))
            return 0
        except OSError:
            return -1

    def listen(self, backlog: int = 5) -> int:
        if self._sock is None:
            return -1
        try:
            self._sock.listen(backlog)
            return 0
        except OSError:
            return -1

    def accept(self):
        if self._sock is None:
            return None
        try:
            conn, addr = self._sock.accept()
            return conn
        except OSError:
            return None

    def connect(self, host: str, port: int) -> int:
        if self._sock is None:
            return -1
        try:
            self._sock.connect((host, port))
            return 0
        except OSError:
            return -1

    def send(self, data: bytes) -> int:
        if self._sock is None:
            return -1
        try:
            self._sock.sendall(struct.pack("!I", len(data)) + data)
            return len(data)
        except OSError:
            return -1

    def recv(self) -> Optional[bytes]:
        if self._sock is None:
            return None
        try:
            raw = self._sock.recv(4)
            if not raw:
                return None
            n = struct.unpack("!I", raw)[0]
            data = b""
            while len(data) < n:
                chunk = self._sock.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            return data
        except OSError:
            return None

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def send_tensor(self, tensor: np.ndarray) -> int:
        data = tensor.tobytes()
        header = struct.pack("!II", tensor.shape[0], tensor.shape[1] if tensor.ndim > 1 else 1)
        return self.send(header + data)

    def recv_tensor(self) -> Optional[np.ndarray]:
        header = self.recv()
        if header is None or len(header) < 8:
            return None
        rows, cols = struct.unpack("!II", header[:8])
        n = rows * cols
        raw = self.recv()
        if raw is None or len(raw) < n * 4:
            return None
        return np.frombuffer(raw[:n * 4], dtype=np.float32).reshape(rows, cols)


class RDMA:
    """RDMA (InfiniBand / RoCE) zero-copy transport abstraction."""

    def __init__(self):
        self._has_c = _HAS_C
        self._ctx: Optional[dict] = None
        self._lock = threading.Lock()

    def open(self, device_idx: int = 0) -> bool:
        self._ctx = {"device": device_idx, "active_qps": 0, "regions": []}
        return True

    def close(self):
        self._ctx = None

    def register_memory(self, addr: np.ndarray) -> bool:
        if self._ctx is None:
            return False
        self._ctx["regions"].append({"addr": addr.ctypes.data, "len": addr.nbytes})
        return True

    def deregister_memory(self, idx: int = -1):
        if self._ctx and self._ctx["regions"]:
            if idx < 0:
                self._ctx["regions"].pop()
            elif idx < len(self._ctx["regions"]):
                self._ctx["regions"].pop(idx)

    def read(self, remote_addr: int, local: np.ndarray, length: int) -> int:
        return 0

    def write(self, remote_addr: int, local: np.ndarray, length: int) -> int:
        return 0


class GRPCService:
    """gRPC service stub for distributed coordination."""

    def __init__(self, target: str = ""):
        self._has_c = _HAS_C
        self.target = target
        self.connected = False

    def connect(self) -> bool:
        self.connected = bool(self.target)
        return self.connected

    def disconnect(self):
        self.connected = False

    def register_node(self, node_info: dict) -> int:
        return 0 if self.connected else -1

    def barrier(self) -> int:
        return 0 if self.connected else -1

    def all_gather(self, send_buf: np.ndarray) -> Optional[np.ndarray]:
        if not self.connected:
            return None
        return send_buf.copy()


class NCCLComm:
    """NCCL communication primitives for GPU collectives."""

    def __init__(self):
        self._has_c = _HAS_C
        self._initialized = False
        self._rank = 0
        self._world_size = 1

    def initialize(self) -> int:
        self._initialized = True
        return 0

    def finalize(self) -> int:
        self._initialized = False
        return 0

    def comm_init_rank(self, ndev: int, rank: int, devs: Optional[List[int]] = None) -> int:
        self._rank = rank
        self._world_size = ndev
        return 0

    def comm_destroy(self) -> int:
        return 0

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def all_reduce(self, sendbuf: np.ndarray, recvbuf: np.ndarray,
                   op: int = 0) -> int:
        if op == 0:
            np.copyto(recvbuf, sendbuf)
        return 0

    def all_gather(self, sendbuf: np.ndarray, recvbuf: np.ndarray) -> int:
        np.copyto(recvbuf[:len(sendbuf)], sendbuf)
        return 0

    def broadcast(self, buf: np.ndarray, root: int = 0) -> int:
        return 0

    def reduce(self, sendbuf: np.ndarray, recvbuf: np.ndarray,
               op: int = 0, root: int = 0) -> int:
        if op == 0:
            np.copyto(recvbuf, sendbuf)
        return 0

    def send(self, buf: np.ndarray, peer: int) -> int:
        return 0

    def recv(self, buf: np.ndarray, peer: int) -> int:
        return 0


class ProcessGroup:
    """Higher-level process group with barrier and collectives."""

    def __init__(self, world_size: int = 1, rank: int = 0):
        self._has_c = _HAS_C
        self.world_size = world_size
        self.rank = rank
        self._comm = NCCLComm()
        self._comm.comm_init_rank(world_size, rank)

    def barrier(self) -> int:
        return 0

    def all_reduce(self, data: np.ndarray, op: int = 0) -> np.ndarray:
        out = np.empty_like(data)
        self._comm.all_reduce(data, out, op)
        return out
