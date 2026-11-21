# Sparse Tensors

SNEPPX-Alg supports sparse tensors in COO, CSR, CSC, and BSR formats.

## Creating sparse tensors

```python
import numpy as np
from SneppX_ALG.interface_bindings.sparse import sparse_coo_tensor, sparse_csr_tensor

# COO format
indices = np.array([[0, 1, 2], [0, 1, 2]])
values = np.array([1.0, 2.0, 3.0])
sp = sparse_coo_tensor(indices, values, (3, 3))
```

## Dense interop

```python
dense = sp.to_dense()
sparse = dense.to_sparse()
```

## Sparse operations

```python
from SneppX_ALG.interface_bindings.sparse import sparse_mm, sparse_addmm

# Sparse @ dense
result = sparse_mm(sparse_tensor, dense_matrix)

# Sparse @ sparse
result = sparse_mm(sparse_a, sparse_b)
```

## Layouts

- `Strided` — dense (default)
- `SparseCOO` — coordinate format
- `SparseCSR` — compressed sparse row
- `SparseCSC` — compressed sparse column
- `SparseBSR` — block sparse row

## SparseAdam optimizer

```python
from SneppX_ALG.interface_bindings.optim import SparseAdam

opt = SparseAdam(model.parameters(), lr=1e-3)
```

SparseAdam handles sparse gradients efficiently by updating only non-zero entries.