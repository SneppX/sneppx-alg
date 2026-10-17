# Migrating from PyTorch to SNEPPX-Algo

A side-by-side mapping from common `torch` APIs to `SneppX_ALG` equivalents.
The SNEPPX Python layer mirrors PyTorch's ergonomics but requires the **C
backend** (`_HAS_C_BACKEND is True`) for training; pure-NumPy fallbacks exist
for inference-grade ops.

## Installation

```bash
# PyTorch
pip install torch
# SneppX
pip install sneppx-alg            # wheels include the C backend
# or from source:
git clone https://github.com/SneppX/sneppx-alg
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Set the path (from source builds):

```powershell
$env:PYTHONPATH = "bindings/python"
```

## Tensor

| PyTorch | SNEPPX-Algo |
|---------|-------------|
| `torch.tensor([...])` | `Tensor([...])` |
| `torch.zeros(4, 8)` | `Tensor.zeros(4, 8)` |
| `torch.randn(4, 8)` | `Tensor.randn(4, 8)` |
| `x.numpy()` | `x.numpy()` |
| `x.to("cuda")` | `x.to("cuda")` |
| `x @ y` / `x.matmul(y)` | `x @ y` (`__matmul__`) |
| `x.requires_grad` | `x.requires_grad` |
| `x.grad` | `x.grad` |

```python
# PyTorch
import torch
x = torch.randn(4, 8)
w = torch.randn(8, 16, requires_grad=True)
y = x @ w
y.sum().backward()

# SNEPPX
from SneppX_ALG import Tensor
x = Tensor.randn((4, 8))
w = Tensor.randn((8, 16), requires_grad=True)
y = x @ w
y.backward()           # autodiff tape; requires C backend
```

## nn.Module ↔ Module

| PyTorch | SNEPPX-Algo |
|---------|-------------|
| `nn.Module` | `Module` |
| `nn.Linear` | `Linear` |
| `nn.Embedding` | `Embedding` |
| `nn.LayerNorm` | `LayerNorm` |
| `nn.RMSNorm` | `RMSNorm` |
| `nn.Dropout` | `Dropout` |
| `nn.Sequential` | `Sequential` |
| `nn.MultiheadAttention` | `MultiheadAttention` |
| `nn.TransformerEncoderLayer` | `TransformerBlock` |
| `nn.Transformer` | `Transformer` |
| `model.parameters()` | `model.parameters()` |
| `model.state_dict()` | `model.state_dict()` |
| `model.load_state_dict(...)` | `model.load_state_dict(...)` |
| `model.to(device)` | `model.to(device)` |
| `model.train()` / `model.eval()` | `model.train()` / `model.eval()` |

```python
# PyTorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.GELU(), nn.Linear(256, 10))
    def forward(self, x):
        return self.net(x.flatten(1))

# SNEPPX
from SneppX_ALG import Module, Linear, Sequential, TransformerBlock
class MLP(Module):
    def __init__(self):
        super().__init__()
        self.net = Sequential(Linear(784, 256), GELU(), Linear(256, 10))
    def forward(self, x):
        return self.net(x.reshape((-1, 784)))
```

## Optimizers

| PyTorch | SNEPPX-Algo |
|---------|-------------|
| `torch.optim.SGD` | `SGD` |
| `torch.optim.AdamW` | `AdamW` |
| (Lion) | `Lion` |
| (LAMB) | `LAMB` |
| `torch.optim.lr_scheduler.CosineAnnealingLR` | `CosineAnnealingLR` |

```python
# PyTorch
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

# SNEPPX
from SneppX_ALG import AdamW, CosineAnnealingLR
opt   = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
sched = CosineAnnealingLR(opt, min_lr=1e-5, max_lr=2e-4, total_steps=100)
```

## Training loop

| PyTorch | SNEPPX-Algo |
|---------|-------------|
| `loss.backward()` | `loss.backward()` |
| `optimizer.step()` | `optimizer.step()` |
| `optimizer.zero_grad()` | `optimizer.zero_grad()` |
| (Trainer) | `Trainer.fit(loader)` |

```python
# SNEPPX — tape-based autodiff via the C backend
from SneppX_ALG import Tensor, AdamW, MSELoss
opt = AdamW(model.parameters(), lr=1e-3)
loss_fn = MSELoss()
for x, y in loader:
    opt.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
```

> The `SneppX_ALG.Trainer` class wraps the C training loop
> (`Trainer.fit`); for low-level control use the optimizer directly as above.

## Data

| PyTorch | SNEPPX-Algo |
|---------|-------------|
| `torch.utils.data.Dataset` | `Dataset` |
| `torch.utils.data.TensorDataset` | `TensorDataset` |
| `torch.utils.data.DataLoader` | `DataLoader` (interface_bindings.data_loader) |
| `torch.utils.data.distributed.DistributedSampler` | `DistributedSampler` |

## Distributed

| PyTorch | SNEPPX-Algo |
|---------|-------------|
| `torch.distributed.init_process_group` | `init_process_group` |
| `torch.nn.parallel.DistributedDataParallel` | `DistributedDataParallel` / `DistributedWrapper` |
| `torch.distributed.launch` / `torchrun` | `launch(train_fn, num_gpus=...)` |
| `torch.distributed.is_initialized` | `DistributedContext.initialized` |

## Generation

| PyTorch (HF `model.generate`) | SNEPPX-Algo |
|-------------------------------|-------------|
| `GenerationConfig(...)` | `GenerationConfig(...)` |
| `model.generate(...)` | `generate(model, input_ids, ...)` |
| `LogitsWarperList` | `top_k_top_p_filtering(...)` |
| `AutoTokenizer` | `Tokenizer` / `SimpleTokenizer` |

```python
from SneppX_ALG.interface_bindings.generation import generate, GenerationConfig
from SneppX_ALG import Tokenizer

gen_config = GenerationConfig(max_new_tokens=64, temperature=0.7, top_p=0.9)
tok = Tokenizer(vocab_size=32000)
ids = tok.encode("Hello, SneppX")
result = generate(model, ids, generation_config=gen_config)
print(tok.decode(result["output_ids"].tolist()[0]))
```

## Quick reference table

| Concept | torch | SneppX-Algo |
|---------|-------|-------------|
| Backend | CUDA | C11/C++20 (`_SNEPPX_c`) + CUDA (`_HAS_CUDA`) |
| Grad | autograd | tape-based `backward()` |
| RNG | `torch.manual_seed` | `seed=` on config |
| Device | `torch.device` | `"cpu"` / `"cuda"` strings |
| Dtype | `torch.float32` | `"float32"`, `Dtype.FLOAT32` |
| Save | `torch.save` | `model.save_checkpoint(path)` |

## Gotchas

- SNEPPX `Tensor` uses **4-space indentation in Python**, `SNEPPX_` prefix on
  C APIs, and `void` parameter lists in C — not a Python concern, but the
  binding docstrings follow the same conventions.
- `Linear.forward` does `x @ weight.T` (matches PyTorch convention).
- Without the C backend, `backward()`/`Trainer.fit` raise `RuntimeError`.
  Build the extension (`cmake --build build --config Release`) to enable them.
