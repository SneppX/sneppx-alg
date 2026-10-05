# Python Framework Capabilities (recent)

## Higher-order autograd

`create_graph=True` is supported through 40+ ops (Conv1d, Linear, LayerNorm,
RMSNorm, MSE/CrossEntropy/KLDiv/NLL/BCE/SmoothL1/Huber losses, activations,
pooling, sparse-related paths). Double-backward Hessians are exact.

Forward/reverse mode helpers live in the autograd module:

```python
from SneppX_ALG.interface_bindings.autograd import jvp, vjp
```

## Functional API

```python
import SneppX_ALG as sx
F = sx.F  # differentiable nn.functional
```

~70 ops: activations, convs (1d/2d/3d, transposed), pooling (incl. adaptive),
norms (batch/group/instance/layer/rms), padding, unfold/fold, embedding ops,
losses, interpolate, pixel/channel shuffle, one_hot, pad_sequence.

## nn modules

Conv1d/Conv3d, ConvTranspose1d/2d/3d, MaxPool1d/2d/3d, AvgPool1d/2d/3d,
adaptive pools, InstanceNorm1d/2d/3d, Upsample, Unfold/Fold, MaxUnpool1d/2d/3d,
EmbeddingBag, ChannelShuffle, PixelShuffle/Unshuffle, Bilinear, Flatten,
ParameterList, ParameterDict.

## Optimizers

SGD, Adam, AdamW, NAdam, Rprop, ASGD, LBFGS, SparseAdam.

## Sparse tensors

`SparseTensor` with COO/CSR/CSC/BSR layouts, `to_sparse()` / `to_dense()`,
`sparse_mm`, `sparse_addmm`, `sparse_conv2d`.

## Serialization

`Module.load_state_dict(..., strict=True)` validates keys and shapes;
`register_buffer(name, tensor, persistent=...)`.

## Distributed C kernels

`kernel/distributed/*.c` (ddp, tensor_parallel, expert_parallel, zero,
gradient_comm, pipeline) create and destroy a `SNEPPX_ProcessGroup` per
collective; the NCCL binding falls back to a correct CPU path single-process.
