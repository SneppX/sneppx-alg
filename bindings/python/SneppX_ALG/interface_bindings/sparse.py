"""Sparse tensor primitives (COO / CSR / CSC / BSR) with dense interop.

Provides a ``SparseTensor`` format and factory/ops compatible with the
PyTorch sparse API subset: ``sparse_coo_tensor``, ``sparse_csr_tensor``,
``sparse_csc_tensor``, ``sparse_bsr_tensor``, ``sparse_mm``, ``sparse_addmm``,
``sparse_sum`` and layout constants ``Strided``/``SparseCOO``/``SparseCSR``/
``SparseCSC``/``SparseBSR``.

Design: a single canonical COO storage (int64 ``indices`` of shape ``(2, nnz)``
plus dense ``values``) is kept internally; CSR/CSC/BSR constructors validate
their inputs and derive the COO view. All ops run in NumPy and return dense
``Tensor`` or ``SparseTensor`` results. Additive-only: nothing here touches the
existing ``Tensor``/``nn`` code, which receives a lazy ``to_sparse`` hook.
"""

import numpy as np

from .tensor import Tensor, _numpy_dtype, _resolve_dtype

# Layout constants (mirror torch.sparse_* ints; strided = dense).
Strided = 0
SparseCOO = 1
SparseCSR = 2
SparseCSC = 3
SparseBSR = 4

_LAYOUT_TO_S = {
    Strided: "strided",
    SparseCOO: "sparse_coo",
    SparseCSR: "sparse_csr",
    SparseCSC: "sparse_csc",
    SparseBSR: "sparse_bsr",
}
_S_TO_LAYOUT = {v: k for k, v in _LAYOUT_TO_S.items()}


def _layout_to_str(layout) -> str:
    if isinstance(layout, str):
        return layout
    return _LAYOUT_TO_S.get(layout, "sparse_coo")


class SparseTensor:
    """COO-backed sparse tensor (canonical storage for all layouts)."""

    def __init__(self, indices, values, size, layout=SparseCOO, dtype="float32",
                 device="cpu", crow=None, col=None, _verified=False):
        super().__init__()
        self.device = device
        self.dtype = _resolve_dtype(dtype)
        self.requires_grad = False
        self.grad = None
        self.size = tuple(int(s) for s in size)
        self.ndim = len(self.size)
        self.layout = _layout_to_str(layout)
        self._layout_const = layout
        self._crow = crow
        self._col = col

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=_numpy_dtype(self.dtype))
        if indices.ndim != 2 or indices.shape[0] != self.ndim:
            raise ValueError(
                f"indices must have shape ({self.ndim}, nnz); got {indices.shape}"
            )
        if indices.shape[1] != values.shape[0]:
            raise ValueError(
                f"nnz mismatch: indices {indices.shape[1]} != values {values.shape[0]}"
            )
        if self.ndim == 2:
            if indices.shape[1] and (indices[0].max() >= self.size[0] or
                                      indices[1].max() >= self.size[1]):
                raise ValueError("index out of bounds for sparse tensor size")
        elif indices.shape[1] and (indices.max() >= max(self.size)):
            raise ValueError("index out of bounds for sparse tensor size")
        if not _verified:
            # coalesce (sort + sum duplicates), matching torch.sparse_coo_tensor
            if indices.shape[1] > 0:
                order = np.lexsort(indices)
                indices = indices[:, order]
                values = values[order]
                uniq, inverse = np.unique(indices.T, axis=0, return_inverse=True)
                if len(uniq) < indices.shape[1]:
                    sums = np.zeros(len(uniq), dtype=values.dtype)
                    np.add.at(sums, inverse, values)
                    indices = uniq.T
                    values = sums
        self._indices = indices.reshape(self.ndim, 0) if indices.size == 0 else indices
        self._values = values

    # ---- shape / introspection ----
    @property
    def shape(self):
        return self.size

    @property
    def is_sparse(self) -> bool:
        return True

    @property
    def _nnz(self) -> int:
        return self._indices.shape[1]

    def indices(self):
        """Return a copy of the 2-D long indices tensor."""
        return Tensor(self._indices.copy(), dtype="int64", device=self.device)

    def values(self):
        """Return a copy of the nnz values tensor (1-D)."""
        return Tensor(self._values.copy(), dtype=self.dtype, device=self.device)

    def crow_indices(self):
        """Compressed row indices for CSR/BSR views (computed from COO)."""
        if self._crow is not None:
            return Tensor(np.asarray(self._crow, dtype=np.int64).copy(),
                          dtype="int64", device=self.device)
        n, m = self.size
        crow = np.zeros(n + 1, dtype=np.int64)
        if self._nnz:
            np.add.at(crow, self._indices[0] + 1, 1)
            crow = np.cumsum(crow)
        return Tensor(crow, dtype="int64", device=self.device)

    def col_indices(self):
        """Column indices within each row for CSR/BSR views (computed from COO)."""
        if self._col is not None:
            return Tensor(np.asarray(self._col, dtype=np.int64).copy(),
                          dtype="int64", device=self.device)
        if self._nnz:
            order = np.lexsort((self._indices[1], self._indices[0]))
            cols = self._indices[1, order]
            return Tensor(cols, dtype="int64", device=self.device)
        return Tensor(np.zeros(0, dtype=np.int64), dtype="int64", device=self.device)

    def ccol_indices(self):
        """Compressed column indices for CSC views (computed from COO)."""
        n, m = self.size
        ccol = np.zeros(m + 1, dtype=np.int64)
        if self._nnz:
            np.add.at(ccol, self._indices[1] + 1, 1)
            ccol = np.cumsum(ccol)
        return Tensor(ccol, dtype="int64", device=self.device)

    def row_indices(self):
        """Row indices within each column for CSC views (computed from COO)."""
        if self._nnz:
            order = np.lexsort((self._indices[0], self._indices[1]))
            rows = self._indices[0, order]
            return Tensor(rows, dtype="int64", device=self.device)
        return Tensor(np.zeros(0, dtype=np.int64), dtype="int64", device=self.device)

    @property
    def numel(self) -> int:
        return int(np.prod(self.size)) if self.size else 0

    @property
    def dtype_name(self) -> str:
        return self.dtype

    # ---- dense interop ----
    def to_dense(self):
        """Convert to a dense Tensor."""
        out = np.zeros(self.size, dtype=_numpy_dtype(self.dtype))
        if self.ndim == 2 and self._nnz:
            np.add.at(out, (self._indices[0], self._indices[1]), self._values)
        elif self._nnz:
            flat = np.ravel_multi_index(tuple(self._indices), dims=self.size)
            np.add.at(out.ravel(), flat, self._values)
        return Tensor(out, dtype=self.dtype, device=self.device)

    @classmethod
    def from_tensor(cls, x, layout=SparseCOO):
        """Build a SparseTensor from a dense tensor (keep nonzero entries)."""
        arr = np.asarray(x.data, dtype=_numpy_dtype(x.dtype_name))
        if arr.ndim != 2:
            raise ValueError("to_sparse currently supports 2-D inputs")
        nz = np.nonzero(arr)
        if len(nz[0]) == 0:
            inds = np.zeros((2, 0), dtype=np.int64)
            vals = np.zeros(0, dtype=_numpy_dtype(x.dtype_name))
        else:
            inds = np.stack(nz, axis=0).astype(np.int64)
            vals = arr[inds[0], inds[1]]
        return cls(inds, vals, arr.shape, layout=layout, dtype=x.dtype_name,
                   device=x.device)

    # ---- conversions between register formats ----
    def coalesce(self):
        return SparseTensor(
            self._indices.copy(), self._values.copy(), self.size,
            layout=self._layout_const, dtype=self.dtype, device=self.device,
        )

    def to_sparse_coo(self):
        return self.coalesce()

    def to_sparse_csr(self):
        crow = self.crow_indices()
        col = self.col_indices()
        return SparseTensor(
            self._indices.copy(), self._values.copy(), self.size,
            layout=SparseCSR, dtype=self.dtype, device=self.device,
            crow=np.asarray(crow.data), col=np.asarray(col.data))

    def to_sparse_csc(self):
        ccol = self.ccol_indices()
        row = self.row_indices()
        return SparseTensor(
            self._indices.copy(), self._values.copy(), self.size,
            layout=SparseCSC, dtype=self.dtype, device=self.device,
            crow=np.asarray(ccol.data), col=np.asarray(row.data))

    def to_sparse_bsr(self, blocksize=(1, 1)):
        if self.ndim != 2:
            raise ValueError("BSR supports 2-D tensors")
        br, bc = blocksize
        n, m = self.size
        if n % br or m % bc:
            raise ValueError(
                f"tensor size {self.size} not divisible by blocksize {blocksize}")
        row = self._indices[0] // br
        crow = np.zeros(n // br + 1, dtype=np.int64)
        np.add.at(crow, row + 1, 1)
        crow = np.cumsum(crow)
        order = np.lexsort((self._indices[1] % bc, self._indices[0] % br,
                            self._indices[1], self._indices[0]))
        col_b = np.asarray(self._indices[1][order] // bc, dtype=np.int64)
        return SparseTensor(
            self._indices.copy(), self._values.copy(), self.size,
            layout=SparseBSR, dtype=self.dtype, device=self.device,
            crow=crow, col=col_b)

    # ---- arithmetic ----
    def mm(self, other):
        return sparse_mm(self, other)

    def __matmul__(self, other):
        return sparse_mm(self, other)

    def sum(self, dim=None):
        return sparse_sum(self, dim)

    def mul(self, other):
        if isinstance(other, SparseTensor):
            return self._mul_sparse(other)
        arr = np.asarray(other.data) if isinstance(other, Tensor) else np.asarray(other)
        if arr.ndim == 0 or arr.size == 1:
            return SparseTensor(
                self._indices.copy(), self._values * arr.reshape(-1)[0],
                self.size, layout=self._layout_const, dtype=self.dtype,
                device=self.device)
        if isinstance(other, Tensor):
            arr = np.asarray(other.data, dtype=_numpy_dtype(other.dtype_name))
        dense = self.to_dense().data * arr
        return SparseTensor.from_tensor(
            Tensor(dense, dtype=self.dtype, device=self.device),
            layout=self._layout_const)

    def _mul_sparse(self, other):
        # elementwise sparse*sparse: keep intersection (sparse sparsity)
        a_idx = {tuple(self._indices[:, k]): k for k in range(self._nnz)}
        idxs, vals = [], []
        for k in range(other._nnz):
            key = tuple(other._indices[:, k])
            if key in a_idx:
                idxs.append(key)
                vals.append(self._values[a_idx[key]] * other._values[k])
        inds = np.array(idxs, dtype=np.int64).T if idxs else np.zeros((self.ndim, 0), dtype=np.int64)
        varr = np.array(vals, dtype=_numpy_dtype(self.dtype)) if vals else np.zeros(0, dtype=_numpy_dtype(self.dtype))
        return SparseTensor(inds, varr, self.size, layout=SparseCOO,
                            dtype=self.dtype, device=self.device)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __repr__(self):
        return (
            f"SparseTensor(size={self.size}, layout={self.layout}, "
            f"nnz={self._nnz}, dtype={self.dtype})"
        )


# --------------------------------------------------------------------------
# Factory constructors (PyTorch-compatible subset)


def sparse_coo_tensor(indices, values, size=None, dtype=None, device="cpu"):
    """Create a 2-D COO sparse tensor from indices (2, nnz) and values."""
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values)
    if size is None:
        if indices.ndim != 2 or indices.shape[0] != 2:
            raise ValueError("size must be provided for non-2-row indices")
        size = (int(indices[0].max()) + 1, int(indices[1].max()) + 1)
    if dtype is None:
        dtype = _resolve_dtype(str(values.dtype)) if values.size else "float32"
    return SparseTensor(indices, values, size, layout=SparseCOO, dtype=dtype,
                        device=device)


def sparse_csr_tensor(crow_indices, col_indices, values, size=None, dtype=None,
                      device="cpu"):
    """Create a CSR sparse tensor from compressed-row arrays."""
    crow = np.asarray(crow_indices, dtype=np.int64)
    col = np.asarray(col_indices, dtype=np.int64)
    values = np.asarray(values)
    if size is None:
        n = len(crow) - 1
        m = int(col.max()) + 1 if col.size else 0
        size = (n, m)
    n, m = size
    if len(crow) != n + 1 or crow[0] != 0 or crow[-1] != len(col):
        raise ValueError("invalid crow_indices for CSR")
    # expand to (2, nnz)
    rows = np.repeat(np.arange(n), np.diff(crow))
    inds = np.stack([rows, col], axis=0)
    if dtype is None:
        dtype = _resolve_dtype(str(values.dtype)) if values.size else "float32"
    return SparseTensor(inds, values, size, layout=SparseCSR, dtype=dtype,
                        device=device, crow=crow, col=col)


def sparse_csc_tensor(ccol_indices, row_indices, values, size=None, dtype=None,
                      device="cpu"):
    """Create a CSC sparse tensor from compressed-column arrays."""
    ccol = np.asarray(ccol_indices, dtype=np.int64)
    row = np.asarray(row_indices, dtype=np.int64)
    values = np.asarray(values)
    if size is None:
        m = len(ccol) - 1
        n = int(row.max()) + 1 if row.size else 0
        size = (n, m)
    n, m = size
    if len(ccol) != m + 1 or ccol[0] != 0 or ccol[-1] != len(row):
        raise ValueError("invalid ccol_indices for CSC")
    cols = np.repeat(np.arange(m), np.diff(ccol))
    inds = np.stack([row, cols], axis=0)
    if dtype is None:
        dtype = _resolve_dtype(str(values.dtype)) if values.size else "float32"
    return SparseTensor(inds, values, size, layout=SparseCSC, dtype=dtype,
                        device=device, crow=ccol, col=row)


def sparse_bsr_tensor(crow_indices, col_indices, values, size, blocksize,
                      dtype=None, device="cpu"):
    """Create a BSR sparse tensor (block storage) from CSR block indices."""
    crow = np.asarray(crow_indices, dtype=np.int64)
    col = np.asarray(col_indices, dtype=np.int64)
    br, bc = blocksize
    n, m = size
    if n % br or m % bc:
        raise ValueError(f"size {size} not divisible by blocksize {blocksize}")
    nb = len(crow) - 1
    if len(col) != crow[-1]:
        raise ValueError("invalid BSR block column indices")
    blk = np.asarray(values)
    req = crow[-1] * br * bc
    if blk.size != req:
        raise ValueError(
            f"BSR values must hold {req} elements; got {blk.size}")
    blk = blk.reshape(crow[-1], br, bc)
    inds, vals = [], []
    for bi in range(nb):
        for start in range(crow[bi], crow[bi + 1]):
            bj = col[start]
            bvals = blk[start]
            nz = np.nonzero(bvals)
            for (r, c) in zip(*nz):
                inds.append([bi * br + r, bj * bc + c])
                vals.append(bvals[r, c])
    inds = np.array(inds, dtype=np.int64).T if inds else np.zeros((2, 0), dtype=np.int64)
    varr = np.array(vals) if vals else np.zeros(0)
    if dtype is None:
        dtype = _resolve_dtype(str(varr.dtype)) if varr.size else "float32"
    return SparseTensor(inds, varr, size, layout=SparseBSR, dtype=dtype,
                        device=device, crow=crow, col=col)


# --------------------------------------------------------------------------
# Sparse ops (return dense Tensors where torch does)


def sparse_mm(mat1, mat2):
    """Matrix product. mat1 may be SparseTensor or Tensor; mat2 either."""
    if isinstance(mat1, SparseTensor) and isinstance(mat2, SparseTensor):
        d1 = np.asarray(mat1.to_dense().data)
        d2 = np.asarray(mat2.to_dense().data)
        return Tensor(d1 @ d2, dtype=mat1.dtype, device=mat1.device)
    if isinstance(mat2, SparseTensor):
        raise ValueError("second operand of sparse mm must be dense")

    if isinstance(mat1, SparseTensor):
        a = mat1
        if isinstance(mat2, Tensor):
            b = np.asarray(mat2.data, dtype=_numpy_dtype(mat2.dtype_name))
            b_dtype = mat2.dtype_name
            out_device = mat1.device
        else:
            b = np.asarray(mat2)
            b_dtype = _resolve_dtype(str(b.dtype))
            out_device = mat1.device
        n, k = a.size
        if b.ndim == 1:
            if k != b.shape[0]:
                raise ValueError(f"matmul size mismatch: {a.size} vs dense {b.shape}")
            out = np.zeros((n,), dtype=_numpy_dtype(b_dtype))
            r, c = a._indices
            v = a._values
            np.add.at(out, r, v * b[c])
        else:
            if k != b.shape[0]:
                raise ValueError(f"matmul size mismatch: {a.size} vs dense {b.shape}")
            out = np.zeros((n,) + b.shape[1:], dtype=_numpy_dtype(b_dtype))
            r, c = a._indices
            v = a._values
            np.add.at(out, r, (v[:, None] * b[c]).astype(_numpy_dtype(b_dtype)))
        return Tensor(out, dtype=b_dtype, device=out_device)

    # both dense
    if isinstance(mat1, Tensor):
        a_np = np.asarray(mat1.data)
        a_dtype = mat1.dtype_name
        out_device = mat1.device
    else:
        a_np = np.asarray(mat1)
        a_dtype = _resolve_dtype(str(a_np.dtype))
        out_device = "cpu"
    if isinstance(mat2, Tensor):
        b_np = np.asarray(mat2.data)
    else:
        b_np = np.asarray(mat2)
    return Tensor(a_np @ b_np, dtype=a_dtype, device=out_device)


def sparse_addmm(input, mat1, mat2, beta=1.0, alpha=1.0):
    """beta*input + alpha*(mat1 @ mat2); mat1 may be sparse."""
    res = sparse_mm(mat1, mat2)
    res_np = np.asarray(res.data)
    inp = np.asarray(input.data) if isinstance(input, Tensor) else np.asarray(input)
    return Tensor(beta * inp + alpha * res_np,
                  dtype=res.dtype_name, device=res.device)


def sparse_sum(input, dim=None, dtype=None):
    """Sum of a sparse tensor. Single dim -> dense Tensor; None -> scalar."""
    dense = input.to_dense()
    if dtype is not None and dtype != input.dtype:
        dense = Tensor(dense.data.astype(_numpy_dtype(dtype)), dtype=dtype,
                       device=input.device)
    if dim is None:
        return np.sum(np.asarray(dense.data, dtype=dense.dtype_name),
                      dtype=dense.dtype_name if dtype is None else _numpy_dtype(dtype))
    return dense.sum(dim)


def sparse_softmax(input, dim):
    """Softmax over one dimension of a sparse tensor (dense result)."""
    ex = np.exp(np.asarray(input.to_dense().data) -
                np.expand_dims(np.max(np.asarray(input.to_dense().data), axis=dim), axis=dim))
    return Tensor(ex / ex.sum(axis=dim, keepdims=True), dtype=input.dtype,
                  device=input.device)