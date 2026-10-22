# Getting Started

This guide walks you through installing SNEPPX-Alg and running your first training loop.

## Install

```bash
pip install sneppx-alg
```

Or from source:

```bash
git clone https://github.com/SneppX/sneppx-alg.git
cd sneppx-alg
pip install -e .
```

## First tensor

```python
import numpy as np
from SneppX_ALG import Tensor

x = Tensor(np.array([1.0, 2.0, 3.0]))
print(x.shape)  # (3,)
```

## First training loop

```python
import numpy as np
from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings import nn, optim

np.random.seed(0)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)
    def forward(self, x):
        return self.fc2(nn.functional.relu(self.fc1(x)))

model = MLP()
opt = optim.Adam(model.parameters(), lr=1e-2)

for step in range(200):
    x = Tensor(np.random.randn(32, 4).astype(np.float32))
    y = Tensor((2.0 * x.data[:, :2]).astype(np.float32))
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()
    opt.zero_grad()

print("final loss:", round(float(np.asarray(loss.data).reshape(-1)[0]), 4))
```

## Next steps

- [Autograd guide](autograd-guide.md) — higher-order AD, jvp/vjp
- [Distributed guide](distributed-guide.md) — DDP, ZeRO, tensor parallel
- [Security layers](security_layers.md) — S0-S9 overview