# Getting Started

This guide walks you through installing SNEPPX-Alg and running your first training loop.

## Installation

```bash
pip install sneppx-alg
```

Or install in development mode from source:

```powershell
git clone https://github.com/SneppX/sneppx-alg.git
cd sneppx-alg
pip install -e .
```

To run with local development bindings without installation:
```powershell
$env:PYTHONPATH = "bindings/python"
```

## First Tensor & Autograd

Tensors support automatic differentiation out of the box with tape-based backpropagation:

```python
import numpy as np
from SneppX_ALG import Tensor

# Create a tensor requiring gradients
x = Tensor(np.array([2.0, 3.0, 4.0], dtype=np.float32), requires_grad=True)

# Forward arithmetic
y = (x * x + 3.0 * x).sum()

# Backward pass to compute dy/dx
y.backward()

print("x.grad:", x.grad)  # [7.0, 9.0, 11.0]
```

## First Neural Network Training Loop

```python
import numpy as np
from SneppX_ALG import Tensor, F
from SneppX_ALG.interface_bindings import nn, optim

np.random.seed(0)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

model = MLP()
opt = optim.Adam(model.parameters(), lr=1e-2)

for step in range(200):
    x = Tensor(np.random.randn(32, 4).astype(np.float32))
    y = Tensor((2.0 * x.data[:, :2]).astype(np.float32))

    loss = F.mse_loss(model(x), y)
    loss.backward()

    opt.step()
    opt.zero_grad()

print("final loss:", round(float(np.asarray(loss.data).reshape(-1)[0]), 4))
```

## Next Steps

- [Autograd Guide](autograd-guide.md) — higher-order automatic differentiation, JVP and VJP
- [Distributed Guide](distributed-guide.md) — DDP, ZeRO-1/2/3, and tensor parallelism
- [Security Layers](security_layers.md) — S0–S9 AI framework defense-in-depth
- [Model Zoo Browser](model-zoo-browser.md) — browse supported foundation model architectures
