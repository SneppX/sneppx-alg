"""
Example 1: Linear Regression using SNEPPX-Alg PyTorch-Like API
"""
import numpy as np
from SneppX_ALG.interface_bindings.pytorch_api import Tensor, Module, Linear, Adam, Trainer, Dataset, DataLoader

class LinearRegressionDataset(Dataset):
    def __init__(self, n=200):
        self.x = np.random.randn(n, 1).astype(np.float32)
        self.y = (3.5 * self.x + 1.2 + np.random.randn(n, 1) * 0.1).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return Tensor(self.x[idx]), Tensor(self.y[idx])

class LinearRegressionModel(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def mse_loss(preds, targets):
    diff = preds.data - targets.data
    return Tensor(np.mean(diff ** 2), requires_grad=True)

if __name__ == "__main__":
    dataset = LinearRegressionDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = LinearRegressionModel()
    optimizer = Adam(model.parameters(), lr=0.01)
    
    trainer = Trainer(model, mse_loss, optimizer)
    trainer.fit(loader, epochs=5)
    print("Linear Regression trained successfully!")
