"""Tests for the sparse tensor primitives (COO/CSR/CSC/BSR + sparse ops).

Covers: sparse_coo_tensor / sparse_csr_tensor / sparse_csc_tensor /
sparse_bsr_tensor constructors, coalescing, rich-to-dense / dense-to-sparse
roundtrips, sparse_mm (sparse @ dense, sparse @ sparse, dense @ dense),
sparse_addmm, sparse_sum, sparse_softmax, Tensor.to_sparse/to_dense, and
layout constants / top-level wiring.
"""

import sys

sys.path.insert(0, "bindings/python")

import numpy as np

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import sparse
from SneppX_ALG.interface_bindings.sparse import (
    SparseTensor,
    Strided,
    SparseCOO,
    SparseCSR,
    SparseCSC,
    SparseBSR,
    sparse_coo_tensor,
    sparse_csr_tensor,
    sparse_csc_tensor,
    sparse_bsr_tensor,
    sparse_mm,
    sparse_addmm,
    sparse_sum,
    sparse_softmax,
    sparse_conv2d,
)


def test_coo_construction_coalesces():
    inds = np.array([[0, 1, 0], [1, 2, 1]])
    vals = np.array([1.0, 2.0, 3.0])  # duplicate (0,1)
    t = sparse_coo_tensor(inds, vals, size=(2, 3))
    assert t.shape == (2, 3)
    dense = np.asarray(t.to_dense().data)
    assert dense[0, 1] == 4.0  # 1.0 + 3.0 coalesced
    assert dense[1, 2] == 2.0
    assert t._nnz == 2  # duplicate removed


def test_coo_to_dense_and_from_tensor():
    arr = np.array([[0.0, 0.0, 5.0], [0.0, 7.0, 0.0], [0.0, 0.0, 0.0]])
    dense = Tensor(arr.astype("float32"))
    sp = dense.to_sparse("sparse_coo")
    assert sp.layout == "sparse_coo"
    assert sp.is_sparse
    rt = np.asarray(sp.to_dense().data)
    assert np.allclose(rt, arr)
    assert dense.to_dense() is not None
    assert sp.numel == 9


def test_csr_construction_and_fields():
    crow = np.array([0, 2, 3, 5])
    col = np.array([0, 2, 2, 0, 1])
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t = sparse_csr_tensor(crow, col, vals, size=(3, 3))
    assert t.layout == "sparse_csr"
    assert t._nnz == 5
    assert np.asarray(t.crow_indices().data).tolist() == [0, 2, 3, 5]
    dense = np.asarray(t.to_dense().data)
    expected = np.array(
        [[1, 0, 2], [0, 0, 3], [4, 5, 0]], dtype=float
    )
    assert np.allclose(dense, expected)


def test_csc_construction_and_fields():
    ccol = np.array([0, 1, 2, 4])
    row = np.array([0, 1, 0, 2])
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    t = sparse_csc_tensor(ccol, row, vals, size=(3, 3))
    assert t.layout == "sparse_csc"
    assert np.asarray(t.ccol_indices().data).tolist() == [0, 1, 2, 4]
    dense = np.asarray(t.to_dense().data)
    expected = np.array(
        [[1, 0, 3], [0, 2, 0], [0, 0, 4]], dtype=float
    )
    assert np.allclose(dense, expected)


def test_bsr_construction():
    # 4x4 with 2x2 blocks; crow=[0,1,3] -> 2 row-blocks, 3 block entries total
    blk = np.zeros((3, 2, 2))
    blk[0] = np.array([[1, 0], [0, 2]])
    blk[1] = np.array([[0, 3], [4, 0]])
    blk[2] = np.array([[5, 0], [0, 6]])
    t = sparse_bsr_tensor(
        np.array([0, 1, 3]), np.array([0, 0, 1]), blk, size=(4, 4),
        blocksize=(2, 2),
    )
    assert t.layout == "sparse_bsr"
    dense = np.asarray(t.to_dense().data)
    expected = np.zeros((4, 4))
    expected[0:2, 0:2] = [[1, 0], [0, 2]]
    expected[2:4, 0:2] = [[0, 3], [4, 0]]
    expected[2:4, 2:4] = [[5, 0], [0, 6]]
    assert np.allclose(dense, expected)


def test_sparse_mm_sparse_dense():
    sp = sparse_coo_tensor(
        np.array([[0, 1, 2], [0, 2, 1]]), np.array([1.0, 2.0, 3.0]),
        size=(3, 3),
    )
    b = Tensor(np.arange(9, dtype=np.float32).reshape(3, 3) + 1)
    out = sp.mm(b)
    dense_a = np.asarray(sp.to_dense().data)
    expected = dense_a @ np.asarray(b.data)
    assert np.allclose(np.asarray(out.data), expected)
    out2 = sparse_mm(sp, b)
    assert np.allclose(np.asarray(out2.data), np.asarray(out.data))


def test_sparse_mm_sparse_sparse_and_dense_dense():
    a = sparse_coo_tensor(
        np.array([[0, 1], [1, 0]]), np.array([2.0, 3.0]), size=(2, 2)
    )
    c = sparse_coo_tensor(
        np.array([[0, 1], [0, 1]]), np.array([5.0, 6.0]), size=(2, 2)
    )
    out = a.mm(c)
    dense_a = np.asarray(a.to_dense().data)
    dense_c = np.asarray(c.to_dense().data)
    assert np.allclose(np.asarray(out.data), dense_a @ dense_c)
    # dense @ dense passes through
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    y = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
    dd = sparse_mm(x, y)
    assert np.allclose(np.asarray(dd.data), np.asarray(x.data) @ np.asarray(y.data))


def test_sparse_addmm():
    sp = sparse_coo_tensor(
        np.array([[0, 1], [1, 0]]), np.array([1.0, 2.0]), size=(2, 2)
    )
    inp = Tensor(np.full((2, 2), 10.0, dtype=np.float32))
    b = Tensor(np.eye(2, dtype=np.float32))
    out = sparse_addmm(inp, sp, b, beta=0.5, alpha=2.0)
    dense_a = np.asarray(sp.to_dense().data)
    expected = 0.5 * np.asarray(inp.data) + 2.0 * (dense_a @ np.asarray(b.data))
    assert np.allclose(np.asarray(out.data), expected)


def test_sparse_sum():
    arr = np.array([[0.0, 0.0, 5.0], [0.0, 7.0, 0.0], [0.0, 0.0, 8.0]])
    sp = Tensor(arr.astype("float32")).to_sparse()
    assert abs(float(np.asarray(sp.sum().data).reshape(-1)[0]) - 20.0) < 1e-6
    row_sums = np.asarray(sp.sum(dim=1).data)
    assert np.allclose(row_sums, arr.sum(axis=1))
    col_sums = np.asarray(sp.sum(dim=0).data)
    assert np.allclose(col_sums, arr.sum(axis=0))


def test_sparse_mul_scalar_and_dense():
    sp = sparse_coo_tensor(
        np.array([[0, 1], [1, 0]]), np.array([2.0, 3.0]), size=(2, 2)
    )
    sc = sp * 3.0
    assert np.allclose(np.asarray(sc.to_dense().data),
                       [[0, 6], [9, 0]])
    d = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    dm = sp.mul(d)
    assert np.allclose(np.asarray(dm.to_dense().data), [[0, 0], [0, 0]])


def test_softmax():
    sp = sparse_coo_tensor(
        np.array([[0, 1], [0, 1]]), np.array([1.0, 2.0]), size=(2, 2)
    )
    sm = sparse.sparse_softmax(sp, dim=0)
    s = np.asarray(sm.data)
    assert np.allclose(s.sum(axis=0), 1.0, atol=1e-6)


def test_layout_constants_and_wiring():
    assert Strided == 0 and SparseCOO == 1 and SparseCSR == 2
    assert SparseCSC == 3 and SparseBSR == 4
    import SneppX_ALG.interface_bindings as sx
    assert sx.SparseTensor is SparseTensor
    assert callable(sx.sparse_coo_tensor)
    assert sx.sparse.sparse_coo_tensor is sparse_coo_tensor
    assert sparse.SparseCOO == 1


def test_tensor_layout_convenience_methods():
    arr = np.array([[0.0, 5.0], [7.0, 0.0]])
    d = Tensor(arr.astype("float32"))
    csr = d.to_sparse_csr()
    assert csr.layout == "sparse_csr"
    csc = d.to_sparse_csc()
    assert csc.layout == "sparse_csc"
    bsr = d.to_sparse_bsr((1, 1))
    assert bsr.layout == "sparse_bsr"
    coo = d.to_sparse_coo()
    assert coo.layout == "sparse_coo"
    for sp in (csr, csc, bsr, coo):
        assert np.allclose(np.asarray(sp.to_dense().data), arr)


def test_sparse_conv2d():
    x = np.zeros((1, 1, 4, 4), dtype=np.float32)
    x[0, 0, 1, 1] = 1.0
    x[0, 0, 2, 3] = 2.0
    sp = Tensor(x).to_sparse()
    w = Tensor(np.array([[[[1.0]]]], dtype=np.float32))  # 1x1 kernel
    out = sparse.sparse_conv2d(sp, w)
    assert isinstance(out, SparseTensor)
    dense = np.asarray(out.to_dense().data)
    assert dense[0, 0, 1, 1] == 1.0 and dense[0, 0, 2, 3] == 2.0
    assert int(out._nnz) == 2  # sparse output keeps nonzero entries only
    # dense input returns dense Tensor
    dout = sparse.sparse_conv2d(Tensor(x), w)
    assert not isinstance(dout, SparseTensor)
    assert np.allclose(np.asarray(dout.data), x)


def test_sparse_adam_updates_only_nonzero_grad_indices():
    from SneppX_ALG.interface_bindings.optim import SparseAdam

    p = Tensor(np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    p.requires_grad_(True)
    sp_grad = sparse_coo_tensor(
        np.array([[0, 1], [0, 1]]), np.array([1.0, 2.0]), size=(2, 2),
        dtype="float32",
    )
    p.grad = sp_grad

    opt = SparseAdam([p], lr=0.1)
    opt.step()
    new = np.asarray(p.data)
    # only (0,0) and (1,1) may change; (0,1) and (1,0) must stay 1.0
    assert new[0, 1] == 1.0 and new[1, 0] == 1.0
    assert new[0, 0] != 1.0 and new[1, 1] != 1.0
    # dense gradient: falls back to dense Adam, all entries move
    p2 = Tensor(np.full((2, 2), 1.0, dtype=np.float32))
    p2.requires_grad_(True)
    p2.grad = Tensor(np.full((2, 2), 1.0, dtype=np.float32))
    SparseAdam([p2], lr=0.1).step()
    assert np.all(np.asarray(p2.data) != 1.0)


# ---- Higher-order AD (create_graph) tests for sparse ops ----


def _sp(size, inds, vals, dtype="float32"):
    return sparse_coo_tensor(np.asarray(inds, dtype=np.int64),
                             np.asarray(vals, dtype=dtype), size=size)


def test_sparse_mm_backward_create_graph():
    sp = _sp((2, 2), [[0, 1], [1, 0]], [2.0, 3.0])
    sp.requires_grad_(True)
    b = Tensor(np.eye(2, dtype=np.float32))
    b.requires_grad_(True)
    out = sparse_mm(sp, b)
    assert out.requires_grad
    out.backward(create_graph=True)
    assert sp.grad is not None
    assert b.grad is not None
    # grad sp = out_grad @ b^T; out_grad = ones for scalar-like output
    expected_sp_grad = np.ones((2, 2)) @ np.eye(2).T
    assert np.allclose(np.asarray(sp.grad.data), expected_sp_grad)


def test_sparse_mm_second_order():
    sp = _sp((2, 2), [[0, 1], [1, 0]], [2.0, 3.0])
    sp.requires_grad_(True)
    b = Tensor(np.eye(2, dtype=np.float32))
    b.requires_grad_(True)
    out = sparse_mm(sp, b)
    # first backward with create_graph
    grad1 = out.backward(create_graph=True)
    # second backward on grad sp
    assert sp.grad is not None
    sp.grad.backward(create_graph=True)
    # After second backward, grad_fn graph should still be connected
    assert sp.grad.grad_fn is not None or sp.grad.requires_grad


def test_sparse_addmm_backward_create_graph():
    sp = _sp((2, 2), [[0, 1], [1, 0]], [1.0, 2.0])
    sp.requires_grad_(True)
    inp = Tensor(np.full((2, 2), 10.0, dtype=np.float32))
    inp.requires_grad_(True)
    b = Tensor(np.eye(2, dtype=np.float32))
    b.requires_grad_(True)
    out = sparse_addmm(inp, sp, b, beta=0.5, alpha=2.0)
    assert out.requires_grad
    out.backward(create_graph=True)
    assert sp.grad is not None
    assert inp.grad is not None
    assert b.grad is not None


def test_sparse_sum_backward_create_graph():
    sp = _sp((2, 2), [[0, 1], [1, 0]], [2.0, 3.0])
    sp.requires_grad_(True)
    out = sparse_sum(sp)
    out.backward(create_graph=True)
    assert sp.grad is not None
    # d(sum)/d(values) = 1 for each value
    assert np.allclose(np.asarray(sp.grad.data), [1.0, 1.0])


def test_sparse_sum_backward_dim_create_graph():
    sp = _sp((2, 2), [[0, 1], [1, 0]], [2.0, 3.0])
    sp.requires_grad_(True)
    out = sparse_sum(sp, dim=0)
    out.backward(create_graph=True)
    assert sp.grad is not None
    # sum over dim 0: grad w.r.t. values[i] = grad_output[index[1][i]]
    expected = np.array([1.0, 1.0])  # grad_output is ones, all indices covered
    assert np.allclose(np.asarray(sp.grad.data), expected)


def test_sparse_softmax_backward_create_graph():
    sp = _sp((2, 2), [[0, 1], [0, 1]], [1.0, 2.0])
    sp.requires_grad_(True)
    out = sparse_softmax(sp, dim=0)
    assert out.requires_grad
    out.backward(create_graph=True)
    assert sp.grad is not None
    assert sp.grad.data.shape == (2,)


def test_sparse_conv2d_backward_create_graph():
    x = np.zeros((1, 1, 4, 4), dtype=np.float32)
    x[0, 0, 1, 1] = 1.0
    x[0, 0, 2, 3] = 2.0
    sp = Tensor(x).to_sparse()
    sp.requires_grad_(True)
    w = Tensor(np.array([[[[1.0]]]], dtype=np.float32))
    w.requires_grad_(True)
    out = sparse_conv2d(sp, w)
    assert out.requires_grad
    out.backward(create_graph=True)
    assert sp.grad is not None
    assert w.grad is not None


def test_sparse_ops_leaf_grad_accumulation():
    sp = _sp((2, 2), [[0, 1], [1, 0]], [2.0, 3.0])
    sp.requires_grad_(True)
    b = Tensor(np.eye(2, dtype=np.float32))
    out1 = sparse_mm(sp, b)
    out2 = sparse_mm(sp, b)
    loss = (out1 + out2).sum()
    loss.backward(create_graph=True)
    assert sp.grad is not None
    # gradient should be accumulated from both paths
    expected = np.ones((2, 2)) @ np.eye(2).T * 2.0
    assert np.allclose(np.asarray(sp.grad.data), expected)