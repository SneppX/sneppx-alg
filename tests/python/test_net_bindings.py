"""Tests for net_bindings.py — network infrastructure."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

import numpy as np
from SneppX_ALG.interface_bindings.net_bindings import (
    Topology, TopologyType, SocketComm, RDMA, GRPCService, NCCLComm, ProcessGroup
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
