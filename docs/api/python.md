# Python API Reference

This reference describes the public Python API. The package currently uses a
NumPy-backed implementation for much of the public `Tensor` / `nn` surface;
the optional compiled C backend is exposed through separate bridge APIs and
selected wrappers where native implementations are available. Backend coverage
is not uniform, so consult each component's status before assuming native
execution or production readiness.

> **Security note (S0 — Post-Quantum Crypto).** The PQC implementations are
> experimental and are not yet fully interoperable with the finalized FIPS
> standards. The official ACVP report in [`docs/pqc_kat.md`](../pqc_kat.md)
> records SHA3-256 agreement, ML-KEM key-generation checks, and the current
> ML-DSA / SLH-DSA signature-format gaps. Do not treat these implementations as
> production-certified or use them as the sole protection for sensitive data.
> Release/update signing currently uses Ed25519, not ML-DSA.

## Installation

```powershell
$env:PYTHONPATH = "bindings/python"
python -c "from SneppX_ALG import *"
```

No C compilation needed — wrappers are pure Python with optional C backend.

## Package Structure

```
SneppX_ALG/
  __init__.py                   # Re-exports all public classes
  interface_bindings/
    __init__.py                  # Per-algorithm imports
    tensor.py                    # Tensor class (C-backed or pure NumPy)
    nn.py                        # Linear, Sequential layers
    train.py                     # TrainConfig, CUDA optimizer flag
    optim.py                     # Optimizer wrapper (SGD/AdamW)
    data.py                      # Data pipeline (TextDataset, BPE)
    algo_arc.py                  # ARCLayer, ARCAdversarialTrainGraph
    algo_npe.py                  # NPEInstruction, NPEProgram, NPECompiler, NPEVM
    algo_fm.py                   # FMController, FMSyncNCCL
    algo_hss.py                  # HSSModel
    algo_ser.py                  # SERModel
    checkpoint.py                # Checkpoint coordinator, fault tolerance
    profiler.py                  # Profiler, Timer decorator
    model_zoo.py                 # from_pretrained(), model configs
    quantization.py              # QuantMode, QuantizedLinear
```

## Core Types

### Tensor

```python
from SneppX_ALG import Tensor

t = Tensor(np.random.randn(4, 8).astype(np.float32))
arr = t.numpy()          # -> np.ndarray
val = t.item()            # -> float
t2 = Tensor.zeros(4, 8)
t3 = Tensor.ones(4, 8)
t4 = Tensor.randn(4, 8)
t5 = Tensor.from_numpy(np.array(...))

# Operator overloads
c = a + b      # __add__
c = a - b      # __sub__
c = a * b      # __mul__
c = a / b      # __truediv__
c = a @ b      # __matmul__
c = -a         # __neg__
```

### Model

```python
from SneppX_ALG import Model

model = Model({'input_dim': 8, 'output_dim': 8})
out = model.forward(np.random.randn(1, 4, 8).astype(np.float32))
model.train()
model.eval()
```

## Algorithm Wrappers

### ARC

```python
from SneppX_ALG.interface_bindings.algo_arc import ARCLayer, ARCAdversarialTrainGraph

# Defense layer
layer = ARCLayer(input_dim=16, output_dim=16)
output = layer.forward(input_array)
adversarial = layer.simulate_attack(input_array, attack_type=1, epsilon=0.1)

# Adversarial training graph
builder = ARCAdversarialTrainGraph(attack_epsilon=0.1)
clean_out, adv_out = builder.build(weights, x_clean)
```

### NPE + JIT

```python
from SneppX_ALG.interface_bindings.algo_npe import NPECompiler, NPEProgram, NPEVM

# Compile a program
compiler = NPECompiler()
prog = compiler.compile([])
opt = compiler.jit_optimize(prog)   # DCE + matmul+relu fusion

# Execute in VM
vm = NPEVM()
vm.load_program(opt)
output = vm.run(input_array)
```

### FM + NCCL

```python
from SneppX_ALG.interface_bindings.algo_fm import FMController, FMSyncNCCL

ctrl = FMController(num_nodes=4, memory_dim=64, memory_capacity=100)
output = ctrl.forward(node_id=0, input_array)

# NCCL sync with callback
nccl = FMSyncNCCL()
def my_callback(data, ctx):
    return data
result = nccl.sync(data_array, callback=my_callback)
```

### SER

```python
from SneppX_ALG.interface_bindings import SERModel

ser = SERModel(num_experts=8, num_active=2, input_dim=32, expert_dim=64, output_dim=32)
output = ser.forward(input_array)
params = ser.parameters()
```

### HSS

```python
from SneppX_ALG.interface_bindings import HSSModel

hss = HSSModel(state_dim=16, input_dim=8, output_dim=8)
output = hss.forward(input_array)
```

## Training

```python
from SneppX_ALG.interface_bindings.train import TrainConfig
from SneppX_ALG import Trainer

# CPU training
config = TrainConfig()
config.learning_rate = 0.01
config.use_cuda_optimizer = False   # default

# CUDA optimizer
config.use_cuda_optimizer = True    # requires SNEPPX_HAS_CUDA

model = Model({'input_dim': 8, 'output_dim': 8})
trainer = Trainer(model, config.__dict__)
loss = trainer.train_step(input_data, target_data)
avg_loss = trainer.evaluate(input_data, target_data)
```

## Utilities

### Optimizer

```python
from SneppX_ALG import Optimizer

opt = Optimizer(params, lr=0.001, optimizer_type='adamw', weight_decay=0.01)
opt.step()
opt.zero_grad()
```

### Sequential / Linear

```python
from SneppX_ALG import Sequential, Linear

net = Sequential(
    Linear(8, 32),
    Linear(32, 16),
)
out = net(np.random.randn(4, 8))
```

### Checkpoint

```python
from SneppX_ALG.interface_bindings.checkpoint import CheckpointWriter, CheckpointReader

writer = CheckpointWriter("checkpoint.bin")
writer.save({"weights": w, "step": 1000})

reader = CheckpointReader("checkpoint.bin")
data = reader.load()
```

### Profiler

```python
from SneppX_ALG.interface_bindings.profiler import Profiler, timeit

profiler = Profiler()
with profiler.profile("forward"):
    out = model.forward(data)

@timeit(profiler)
def my_func():
    pass
```

### Quantization

```python
from SneppX_ALG.interface_bindings.quantization import QuantMode, quantize, dequantize, QuantizedLinear

qlayer = QuantizedLinear(64, 128, mode=QuantMode.INT8_SYM)
q_out = qlayer.forward(data)
```

### Model Zoo

```python
from SneppX_ALG.interface_bindings.model_zoo import (
    get_model_config, from_pretrained, LLMConfig
)

cfg = get_model_config("llama2-7b")
model = from_pretrained("meta-llama/Llama-2-7b")

# LLMConfig constructors
cfg = LLMConfig.from_name("llama3", "8B")
cfg = LLMConfig.from_json('{"family": "llama3", ...}')

# Serialize
json_str = cfg.to_json()

# Extend context to 128K
cfg.extend_context(131072)

# MHA forward pass
output = cfg.forward_mha(hidden_states, attention_mask, position_ids)
```

## Autograd & Higher-Order AD

The `SneppX_ALG.interface_bindings.autograd_ops` module provides a tape-based
autograd engine with support for first- and higher-order derivatives via
`create_graph=True` on `Tensor.backward()`.

```python
from SneppX_ALG.interface_bindings import autograd_ops as ops
from SneppX_ALG import Tensor
import numpy as np

x = Tensor(np.random.randn(4, 4).astype(np.float32))
x.requires_grad_(True)

# First-order backward (default)
out = ops.Relu.apply(x)
out.backward()                      # grad w.r.t. x, no graph retained

# Higher-order backward (create_graph retains the graph)
out.backward(create_graph=True)     # grad w.r.t. x, graph retained for 2nd order
x.grad.backward(create_graph=True)  # second-order gradient
```

### create_graph support

Most element-wise, shape, normalization, and loss ops support `create_graph=True`
in their `backward()` method, enabling second-order and higher derivatives:

**Supported with graph-connected gradients:** Add, Sub, Mul, Div, MatMul,
Pow, Neg, Relu, Sigmoid, Tanh, Gelu, Swish, Softplus, Softsign, Hardswish,
Hardtanh, Mish, SELU, Hardsigmoid, Abs, Sqrt, Exp, Log, SmoothL1Loss, HuberLoss,
BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, NLLLoss, KLDivLoss, LayerNorm,
RMSNorm, DropoutFn, LinearFn, Sqrt, Reciprocal, Clamp, Remainder, Sign, Log1p,
Expm1, Trace, Diagonal, Square, Reshape, Transpose, Permute, Expand, Squeeze,
Unsqueeze, GetItem, Stack, Cat, Mean, Sum, Prod, MinDim, MaxDim, LogSumExp,
GLU, PReluFn, LogSigmoidFn, Softmax, LogSoftmax, SparseMM, SparseAddMM,
SparseSum, SparseSoftmax, SparseConv2d.

**Graph-connected but falls back to NumPy for complex indexing:** Conv1d, Conv2d,
Conv3d, ConvTranspose2d, ConvTranspose3d, MaxPool2d, AdaptiveAvgPoolFn,
AdaptiveMaxPoolFn, MaxUnpoolFn, UpsampleNearestFn, UpsampleBilinearFn,
UpsampleLinearNdFn, UnfoldFn, FoldFn, PadNdFn, EmbeddingFn, EmbeddingBagFn,
BilinearFn, FocalLoss.

For ops without graph-connected backward, calling `backward(create_graph=True)`
produces a NumPy-computed gradient wrapped as a Tensor — first-order values are
correct, but second-order AD will not propagate through that op.

### Sparse autograd

Sparse tensor operations (`SparseMM`, `SparseAddMM`, `SparseSum`, `SparseSoftmax`,
`SparseConv2d`) support `create_graph=True` with graph-connected gradient routing
via scatter-matrix multiplication, enabling second-order AD through sparse
networks.


## Running Tests

The full Python suite uses `pytest` and runs against the same code paths exercised
by the C-backed and NumPy-fallback engines:

```powershell
$env:PYTHONPATH = "bindings/python"
python -m pytest tests/python -q
```

Cryptographic regression and interoperability tests are available. The
self-generated regression suite is distinct from the official ACVP sample
vectors; neither test suite constitutes a FIPS validation certificate. The
official-vector harness currently reports incomplete ML-DSA / SLH-DSA
interoperability. See [the PQC KAT report](../pqc_kat.md) before relying on
these experimental algorithms.

```powershell
$env:PYTHONPATH = "bindings/python"
python -m pytest tests/python/test_crypto_kat.py -q
python -m pytest tests/python/test_acvp_sha3.py tests/python/test_acvp_mlkem.py tests/python/test_acvp_mldsa.py tests/python/test_acvp_slhdsa.py -q
```

## Extended utilities

A set of pure-Python, NumPy-backed utility packages is included under SneppX_ALG:
sampling, 
egularization, callbacks, linalg, signal, graph, stats,
structured_output, 
ag, gents, observability, speculative. Each module
provides minimal, testable primitives and is re-exported from the top-level package.

`python
from SneppX_ALG import agents, rag, structured_output, observability, speculative
`
