"""Tests for algo_fm.py — FM algorithm bindings."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

import numpy as np
from SneppX_ALG.interface_bindings import FMController, FMMemoryBank, FMNode, FMSync


def test_controller_write_read():
    c = FMController(memory_size=128)
    data = np.array([1.0, 2.0, 3.0])
    c.write("key1", data)
    result = c.read("key1")
    assert result is not None
    assert np.allclose(result, data)


def test_controller_missing_key():
    c = FMController()
    assert c.read("nonexistent") is None


def test_controller_clear():
    c = FMController()
    c.write("a", np.array([1]))
    c.write("b", np.array([2]))
    c.clear()
    assert c.read("a") is None
    assert c.read("b") is None


def test_memory_bank_store_retrieve():
    bank = FMMemoryBank(capacity=10)
    emb = np.random.randn(16).astype(np.float32)
    idx = bank.store(emb)
    assert idx == 0
    retrieved = bank.retrieve(idx)
    assert retrieved is not None


def test_memory_bank_capacity():
    bank = FMMemoryBank(capacity=3)
    for i in range(5):
        bank.store(np.array([float(i)]))
    assert bank.size == 3


def test_memory_bank_invalid_idx():
    bank = FMMemoryBank()
    assert bank.retrieve(999) is None


def test_node_process():
    node = FMNode("n1", dim=16)
    x = np.random.randn(4, 16).astype(np.float32)
    out = node.process(x)
    assert out.shape == x.shape


def test_node_route():
    node = FMNode("n1")
    x = np.array([1.0, 2.0])
    routes = node.route(x, ["a", "b"])
    assert "a" in routes
    assert "b" in routes
    assert np.allclose(routes["a"], x)


def test_sync_all_reduce():
    t = np.array([1.0, 2.0, 3.0])
    result = FMSync.all_reduce(t, "sum")
    assert np.allclose(result, t)


def test_sync_federated_avg():
    tensors = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    avg = FMSync.federated_avg(tensors)
    assert np.allclose(avg, [2.0, 3.0])


def test_sync_federated_avg_empty():
    avg = FMSync.federated_avg([])
    assert avg.size == 0


if __name__ == "__main__":
    test_controller_write_read()
    test_controller_missing_key()
    test_controller_clear()
    test_memory_bank_store_retrieve()
    test_memory_bank_capacity()
    test_memory_bank_invalid_idx()
    test_node_process()
    test_node_route()
    test_sync_all_reduce()
    test_sync_federated_avg()
    test_sync_federated_avg_empty()
    print("All algo_fm tests passed.")
