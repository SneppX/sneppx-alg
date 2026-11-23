"""Tests for net_bindings.py — network infrastructure."""

import sys, os
import struct

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python")
)

import numpy as np
import pytest
from SneppX_ALG.interface_bindings import net_bindings as nb
from SneppX_ALG.interface_bindings.net_bindings import (
    Topology,
    TopologyType,
    SocketComm,
    RDMA,
    GRPCService,
    NCCLComm,
    ProcessGroup,
)


def test_topology_ring():
    t = Topology.create_ring(4)
    assert t.world_size == 4
    assert t.get_prev(0) == 3
    assert t.get_next(0) == 1
    assert t.get_prev(2) == 1
    assert t.get_next(2) == 3
    path = t.compute_route(0, 3)
    assert path == [0, 1, 2, 3] or path == [0, 3]


def test_topology_tree():
    t = Topology.create_tree(7)
    assert t.world_size == 7
    assert t.get_parent(0) == -1
    assert t.get_parent(1) == 0
    assert t.get_parent(2) == 0
    assert t.get_children(0) == [1, 2]
    assert t.get_children(1) == [3, 4]


def test_socket_api():
    s = SocketComm()
    s.create()
    assert s._sock is not None
    assert s.send(b"test") == -1  # no connection
    assert s.recv() is None
    s.close()
    assert s._sock is None


def test_socket_lifecycle():
    s = SocketComm()
    s.create()
    assert s._sock is not None
    s.close()
    assert s._sock is None


def test_socket_send_recv_empty():
    s = SocketComm()
    s.create()
    assert s.send(b"") == -1  # no connection
    assert s.recv() is None


def test_rdma_lifecycle():
    r = RDMA()
    assert r.open(0) is True
    r.close()
    assert r._ctx is None


def test_rdma_memory():
    r = RDMA()
    r.open(0)
    arr = np.zeros(1024, dtype=np.float32)
    assert r.register_memory(arr) is True
    r.deregister_memory()
    assert len(r._ctx["regions"]) == 0
    r.close()


def test_grpc_service():
    g = GRPCService("localhost:50051")
    assert g.connect() is True
    assert g.barrier() == 0
    g.disconnect()
    assert g.connected is False
    assert g.barrier() == -1


def test_nccl_comm():
    nccl = NCCLComm()
    assert nccl.initialize() == 0
    nccl.comm_init_rank(4, 0)
    assert nccl.rank == 0
    assert nccl.world_size == 4
    a = np.ones(16, dtype=np.float32)
    b = np.zeros(16, dtype=np.float32)
    nccl.all_reduce(a, b, op=0)
    assert np.allclose(b, a)
    nccl.finalize()


def test_process_group():
    pg = ProcessGroup(world_size=4, rank=1)
    assert pg.rank == 1
    assert pg.world_size == 4
    assert pg.barrier() == 0
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = pg.all_reduce(data)
    assert np.allclose(result, data)


def _connected_pair():
    """Return (listener, client, server) SocketComm objects on loopback."""
    listener = nb.SocketComm()
    listener.create()
    assert listener.bind(0) == 0
    host, port = listener._sock.getsockname()[:2]
    assert listener.listen(1) == 0
    client = nb.SocketComm()
    client.create()
    assert client.connect(host, port) == 0
    conn = listener.accept()
    assert conn is not None
    server = nb.SocketComm()
    server._sock = conn
    return listener, client, server


def test_bind_defaults_to_loopback():
    s = nb.SocketComm()
    s.create()
    try:
        assert s.bind(0) == 0
        assert s._sock.getsockname()[0] == "127.0.0.1"
    finally:
        s.close()


def test_send_recv_roundtrip():
    listener, client, server = _connected_pair()
    try:
        assert client.send(b"hello") == 5
        assert server.recv() == b"hello"
    finally:
        client.close()
        server.close()
        listener.close()


def test_send_recv_tensor_roundtrip():
    listener, client, server = _connected_pair()
    try:
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        assert client.send_tensor(arr) == 8 + arr.nbytes
        out = server.recv_tensor()
        assert out is not None and out.shape == (2, 3)
        assert np.allclose(out, arr)
    finally:
        client.close()
        server.close()
        listener.close()


def test_send_rejects_oversized_frame(monkeypatch):
    monkeypatch.setattr(nb, "MAX_FRAME_BYTES", 8)
    listener, client, server = _connected_pair()
    try:
        assert client.send(b"x" * 9) == -1
    finally:
        client.close()
        server.close()
        listener.close()


def test_recv_rejects_oversized_frame(monkeypatch):
    monkeypatch.setattr(nb, "MAX_FRAME_BYTES", 8)
    listener, client, server = _connected_pair()
    try:
        # Declare a 9-byte frame but send no payload: recv must reject on the
        # declared length alone instead of blocking for the payload.
        client._sock.sendall(struct.pack("!I", 9))
        assert server.recv() is None
    finally:
        client.close()
        server.close()
        listener.close()


def test_recv_tensor_rejects_huge_shape():
    listener, client, server = _connected_pair()
    try:
        # rows*cols would overflow any sane buffer; must be rejected before
        # multiplication/allocation.
        client._sock.sendall(
            struct.pack("!I", 8) + struct.pack("!II", 0xFFFFFFFF, 0xFFFFFFFF)
        )
        assert server.recv_tensor() is None
    finally:
        client.close()
        server.close()
        listener.close()


def test_recv_tensor_rejects_zero_shape():
    listener, client, server = _connected_pair()
    try:
        client._sock.sendall(struct.pack("!I", 8) + struct.pack("!II", 0, 5))
        assert server.recv_tensor() is None
    finally:
        client.close()
        server.close()
        listener.close()


def test_recv_times_out_without_data(monkeypatch):
    monkeypatch.setattr(nb, "DEFAULT_SOCKET_TIMEOUT", 0.2)
    listener, client, server = _connected_pair()
    try:
        import time

        start = time.monotonic()
        assert server.recv() is None
        assert time.monotonic() - start < 5
    finally:
        client.close()
        server.close()
        listener.close()


if __name__ == "__main__":
    test_topology_ring()
    test_topology_tree()
    test_socket_api()
    test_socket_lifecycle()
    test_socket_send_recv_empty()
    test_rdma_lifecycle()
    test_rdma_memory()
    test_grpc_service()
    test_nccl_comm()
    test_process_group()
    print("All net_bindings tests passed.")
