# Python API Reference

Stable as of **v1.1.1**. All algorithm wrappers are complete and delegate to the
compiled C backend (`_SNEPPX_c` / `_arix_c`) when available, falling back to a
pure-NumPy engine otherwise (exposed via the `_HAS_C` flag).

> **Security note (S0 — Post-Quantum Crypto).** The Dilithium / ML-DSA
> implementation in `security/crypto/c/dilithium.c` is verified against the
> official **FIPS 204** known-answer tests shipped in
> `tests/python/data/kat_vectors.json` (see `tests/python/test_crypto_kat.py`).
> Signing of releases and updates uses these primitives, and the KAT suite runs
> in CI and via `pytest tests/python`.

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

## Running Tests

The full Python suite uses `pytest` and runs against the same code paths exercised
by the C-backed and NumPy-fallback engines:

```powershell
$env:PYTHONPATH = "bindings/python"
python -m pytest tests/python -q
```

Crypto correctness is pinned by known-answer tests:

```powershell
$env:PYTHONPATH = "bindings/python"
python -m pytest tests/python/test_crypto_kat.py -q   # Dilithium (FIPS 204) + SPHINCS+ KAT
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
