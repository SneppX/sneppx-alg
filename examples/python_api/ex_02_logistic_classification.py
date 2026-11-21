"""
Example 2: Binary Logistic Classification using SNEPPX-Alg PyTorch-Like API
"""
import numpy as np
from SneppX_ALG.interface_bindings.pytorch_api import Tensor, Module, Linear, Adam, Trainer, Dataset, DataLoader, accuracy

class BinaryDataset(Dataset):
    def __init__(self, n=300):
        self.x = np.random.randn(n, 4).astype(np.float32)
        logits = np.sum(self.x * np.array([1.5, -2.0, 0.5, 1.0]), axis=-1, keepdims=True)
        self.y = (logits > 0).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return Tensor(self.x[idx]), Tensor(self.y[idx])

class LogisticRegression(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(4, 1)

    def forward(self, x):
        # Sigmoid activation
        logits = self.linear(x)
        return Tensor(1.0 / (1.0 + np.exp(-np.clip(logits.data, -50, 50))), requires_grad=True)

def bce_loss(preds, targets):
    eps = 1e-7
    p = np.clip(preds.data, eps, 1 - eps)
    loss = -np.mean(targets.data * np.log(p) + (1 - targets.data) * np.log(1 - p))
    return Tensor(loss, requires_grad=True)

if __name__ == "__main__":
    dataset = BinaryDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LogisticRegression()
    optimizer = Adam(model.parameters(), lr=0.05)

    trainer = Trainer(model, bce_loss, optimizer)
    trainer.fit(loader, epochs=6)
    print("Logistic Regression trained successfully!")
