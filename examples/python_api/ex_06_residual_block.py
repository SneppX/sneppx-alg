"""
Example 6: Residual Network (ResNet) Building Block using SNEPPX-Alg API
"""
import numpy as np
from SneppX_ALG.interface_bindings.pytorch_api import Tensor, Module, Linear, LayerNorm, Sequential, Adam, Trainer, Dataset, DataLoader

class ResBlock(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = Linear(dim, dim)
        self.ln1 = LayerNorm(dim)
        self.fc2 = Linear(dim, dim)
        self.ln2 = LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        h = self.ln1(self.fc1(x))
        # ReLU-like activation in NumPy
        h_data = np.maximum(0, h.data)
        h = Tensor(h_data, requires_grad=h.requires_grad)
        h = self.ln2(self.fc2(h))
        # Residual addition
        out = h + residual
        return Tensor(np.maximum(0, out.data), requires_grad=out.requires_grad)

class ResNetModel(Module):
    def __init__(self):
        super().__init__()
        self.net = Sequential(
            Linear(8, 16),
            ResBlock(16),
            ResBlock(16),
            Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)

def loss_fn(preds, targets):
    exps = np.exp(preds.data - np.max(preds.data, axis=-1, keepdims=True))
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    n = len(targets)
    targets_arr = np.array(targets)
    loss = -np.mean(np.log(probs[np.arange(n), targets_arr] + 1e-7))
    return Tensor(loss, requires_grad=True)

if __name__ == "__main__":
    x = np.random.randn(50, 8).astype(np.float32)
    y = np.random.randint(0, 2, size=(50,)).astype(np.int64)
    dataset = [(Tensor(x[i]), y[i]) for i in range(50)]
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    model = ResNetModel()
    optimizer = Adam(model.parameters(), lr=0.005)
    
    trainer = Trainer(model, loss_fn, optimizer)
    trainer.fit(loader, epochs=4)
    print("Residual Network trained successfully!")
