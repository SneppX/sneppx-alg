"""Network infrastructure bindings — topology, socket, RDMA, gRPC, NCCL.

Wraps C implementations in ``net/`` with pure-Python fallback.
"""

from typing import Optional, List, Dict
import hashlib
import hmac
import ipaddress
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
                self._nodes.append(
                    {
                        "rank": i,
                        "prev": (i - 1) % self.world_size,
                        "next": (i + 1) % self.world_size,
                        "parent": -1,
                        "children": [],
                    }
                )
        elif self.type == TopologyType.TREE:
            for i in range(self.world_size):
                parent = (i - 1) // 2 if i > 0 else -1
                children = [2 * i + 1, 2 * i + 2]
                children = [c for c in children if c < self.world_size]
                self._nodes.append(
                    {
                        "rank": i,
                        "prev": -1,
                        "next": -1,
                        "parent": parent,
                        "children": children,
                    }
                )

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


# M-16: bound framing so a malicious/broken peer cannot force unbounded
# allocation or an indefinite block. Callers may tighten (never loosen) these.
MAX_FRAME_BYTES = 128 * 1024 * 1024  # 128 MiB per frame
DEFAULT_SOCKET_TIMEOUT = 30.0  # seconds


class SocketComm:
    """TCP / Unix socket communication for point-to-point tensor transfers.

    Frames are ``4-byte big-endian length || payload``. With an authentication
    key set, a truncated HMAC-SHA256 tag is appended and verified with a
    constant-time comparison. Listeners bind to loopback by default, frames
    are size-capped, and socket operations have a timeout so a stalled peer
    cannot block indefinitely.
    """

    _TAG_LEN = 16

    def __init__(
        self,
        max_message_bytes: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        self._has_c = _HAS_C
        self._lock = threading.Lock()
        self._sock: Optional[_socket.socket] = None
        if max_message_bytes is not None and int(max_message_bytes) <= 0:
            raise ValueError("max_message_bytes must be positive")
        self._max_message_bytes = None if max_message_bytes is None else int(max_message_bytes)
        self._timeout = None if timeout is None else float(timeout)
        self._auth_key: Optional[bytes] = None

    def _max_frame_bytes(self) -> int:
        """Return the effective cap; instance caps may only tighten globals."""
        cap = int(MAX_FRAME_BYTES)
        if self._max_message_bytes is not None:
            cap = min(cap, self._max_message_bytes)
        return max(0, cap)

    def _socket_timeout(self) -> float:
        return float(DEFAULT_SOCKET_TIMEOUT if self._timeout is None else self._timeout)

    def set_auth_key(self, key: Optional[bytes]) -> None:
        """Set the shared secret for authenticated frames (None disables it)."""
        if key is not None and len(key) < 16:
            raise ValueError("auth key must be at least 16 bytes")
        self._auth_key = bytes(key) if key is not None else None

    def set_timeout(self, timeout: float) -> None:
        self._timeout = float(timeout)
        if self._sock is not None:
            self._sock.settimeout(self._timeout)

    def create(self, sock_type: int = 0):
        if sock_type == 0 or not hasattr(_socket, "AF_UNIX"):
            family = _socket.AF_INET
        else:
            family = _socket.AF_UNIX
        self._sock = _socket.socket(family, _socket.SOCK_STREAM)
        self._sock.settimeout(self._socket_timeout())

    def bind(self, port: int, host: str = "127.0.0.1") -> int:
        """Bind to ``host``; loopback is the safe default."""
        if self._sock is None:
            return -1
        af_unix = getattr(_socket, "AF_UNIX", None)
        if af_unix is not None and self._sock.family == af_unix:
            try:
                self._sock.bind(str(port))
                return 0
            except (OSError, ValueError):
                return -1
        if not isinstance(port, int) or not 0 <= port <= 65535:
            return -1
        if host is None:
            host = "127.0.0.1"
        try:
            ipaddress.ip_address(host)
        except ValueError:
            return -1
        try:
            self._sock.bind((host, port))
            return 0
        except (OSError, ValueError):
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
            conn.settimeout(self._socket_timeout())
            return conn
        except OSError:
            return None

    def connect(self, host: str, port: int) -> int:
        if self._sock is None:
            return -1
        af_unix = getattr(_socket, "AF_UNIX", None)
        if af_unix is not None and self._sock.family == af_unix:
            try:
                self._sock.connect(str(port))
                return 0
            except (OSError, ValueError):
                return -1
        if not isinstance(port, int) or not 0 <= port <= 65535:
            return -1
        try:
            self._sock.connect((host, port))
            return 0
        except (OSError, ValueError):
            return -1

    def _tag(self, framed: bytes) -> bytes:
        assert self._auth_key is not None
        return hmac.new(self._auth_key, framed, hashlib.sha256).digest()[: self._TAG_LEN]

    def send(self, data: bytes) -> int:
        if self._sock is None:
            return -1
        if not isinstance(data, (bytes, bytearray, memoryview)):
            return -1
        payload = bytes(data)
        if len(payload) > self._max_frame_bytes():
            return -1
        try:
            framed = struct.pack("!I", len(payload)) + payload
            if self._auth_key is not None:
                framed += self._tag(framed)
            self._sock.sendall(framed)
            return len(payload)
        except OSError:
            return -1

    def _recv_exact(self, count: int) -> Optional[bytes]:
        """Read exactly ``count`` bytes or return ``None`` on EOF/timeout."""
        if self._sock is None or count < 0:
            return None
        chunks = []
        remaining = count
        while remaining > 0:
            try:
                chunk = self._sock.recv(remaining)
            except OSError:
                return None
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def recv(self, max_bytes: Optional[int] = None) -> Optional[bytes]:
        """Receive one length-prefixed frame.

        The declared length is validated against the effective cap BEFORE any
        payload is read, so a peer cannot request an oversized allocation.
        """
        if self._sock is None:
            return None
        limit = self._max_frame_bytes()
        if max_bytes is not None:
            try:
                limit = min(limit, int(max_bytes))
            except (TypeError, ValueError):
                return None
        if limit < 0:
            return None
        try:
            raw = self._recv_exact(4)
            if raw is None:
                return None
            n = struct.unpack("!I", raw)[0]
            if n > limit:
                return None
            payload = self._recv_exact(n)
            if payload is None:
                return None
            if self._auth_key is not None:
                tag = self._recv_exact(self._TAG_LEN)
                if tag is None or not hmac.compare_digest(tag, self._tag(raw + payload)):
                    return None
            return payload
        except (OSError, struct.error, ValueError):
            return None

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def send_tensor(self, tensor: np.ndarray) -> int:
        if not isinstance(tensor, np.ndarray) or tensor.ndim > 2:
            return -1
        if tensor.ndim == 0:
            rows, cols = 1, 1
        else:
            rows = int(tensor.shape[0])
            cols = int(tensor.shape[1]) if tensor.ndim > 1 else 1
        if rows <= 0 or cols <= 0:
            return -1
        element_cap = self._max_frame_bytes() // 4
        if element_cap <= 0 or rows > element_cap or cols > element_cap:
            return -1
        n = rows * cols
        if n > element_cap or n * 4 + 8 > self._max_frame_bytes():
            return -1
        data = np.ascontiguousarray(tensor, dtype=np.float32).tobytes()
        header = struct.pack("!II", rows, cols)
        # Single length-prefixed frame: 8-byte shape header + payload, matching
        # recv_tensor() below.
        return self.send(header + data)

    def recv_tensor(self, max_elements: Optional[int] = None) -> Optional[np.ndarray]:
        frame = self.recv()
        if frame is None or len(frame) < 8:
            return None
        rows, cols = struct.unpack("!II", frame[:8])
        # Validate dimensions and the product BEFORE multiplying or allocating,
        # so huge/zero-sized shapes are rejected cheaply.
        if rows == 0 or cols == 0:
            return None
        element_cap = self._max_frame_bytes() // 4
        if max_elements is not None:
            element_cap = min(element_cap, int(max_elements))
        if element_cap <= 0 or rows > element_cap or cols > element_cap:
            return None
        n = rows * cols
        if n > element_cap:
            return None
        payload = frame[8:]
        if len(payload) < n * 4:
            return None
        return np.frombuffer(payload[: n * 4], dtype=np.float32).reshape(rows, cols)


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
    """NCCL communication primitives — reference / single-process fallback.

    .. warning::
       No real cross-rank transfer is implemented. ``all_reduce``/``all_gather``
       are local copies and ``broadcast``/``reduce`` are identity no-ops. For
       ``world_size > 1``, ``send``/``recv`` fail closed (-1) rather than
       claiming an unimplemented transfer succeeded. Use :mod:`distributed`
       with a real backend for genuine multi-rank communication.
    """

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

    def all_reduce(self, sendbuf: np.ndarray, recvbuf: np.ndarray, op: int = 0) -> int:
        """Local identity copy only; no cross-rank reduction is performed."""
        if op == 0:
            np.copyto(recvbuf, sendbuf)
        return 0

    def all_gather(self, sendbuf: np.ndarray, recvbuf: np.ndarray) -> int:
        """Copy into the local buffer only; no cross-rank gather is performed."""
        np.copyto(recvbuf[: len(sendbuf)], sendbuf)
        return 0

    def broadcast(self, buf: np.ndarray, root: int = 0) -> int:
        """No-op: no cross-rank broadcast is performed."""
        return 0

    def reduce(self, sendbuf: np.ndarray, recvbuf: np.ndarray, op: int = 0, root: int = 0) -> int:
        """Local identity copy only; no cross-rank reduction is performed."""
        if op == 0:
            np.copyto(recvbuf, sendbuf)
        return 0

    def send(self, buf: np.ndarray, peer: int) -> int:
        """Fail closed for multi-rank: no transfer backend is available."""
        return -1 if self._world_size > 1 else 0

    def recv(self, buf: np.ndarray, peer: int) -> int:
        """Fail closed for multi-rank: no transfer backend is available."""
        return -1 if self._world_size > 1 else 0


class ProcessGroup:
    """Higher-level process group (single-process reference fallback).

    ``barrier`` is a no-op and ``all_reduce`` is a local identity copy; no
    cross-rank communication occurs for ``world_size > 1``. Use the
    :mod:`distributed` module with a real backend for genuine collectives.
    """

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
