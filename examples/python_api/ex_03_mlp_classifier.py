"""
Example 3: Multi-Layer Perceptron (MLP) Classifier using SNEPPX-Alg API
"""
import numpy as np
from SneppX_ALG.interface_bindings.pytorch_api import Tensor, Module, Linear, LayerNorm, Dropout, Sequential, Adam, Trainer, Dataset, DataLoader

class MLPDataset(Dataset):
    def __init__(self, n=400):
        self.x = np.random.randn(n, 10).astype(np.float32)
        self.y = np.random.randint(0, 3, size=(n,)).astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return Tensor(self.x[idx]), self.y[idx]

class MLPClassifier(Module):
    def __init__(self):
        super().__init__()
        self.net = Sequential(
            Linear(10, 32),
            LayerNorm(32),
            Dropout(0.1),
            Linear(32, 3)
        )

    def forward(self, x):
        return self.net(x)

def ce_loss(preds, targets):
    # Softmax cross entropy
    exps = np.exp(preds.data - np.max(preds.data, axis=-1, keepdims=True))
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    n = targets.shape[0] if isinstance(targets, np.ndarray) else len(targets)
    targets_arr = targets.numpy() if hasattr(targets, 'numpy') else np.array(targets)
    loss = -np.mean(np.log(probs[np.arange(n), targets_arr] + 1e-7))
    return Tensor(loss, requires_grad=True)

if __name__ == "__main__":
    dataset = MLPDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = MLPClassifier()
    optimizer = Adam(model.parameters(), lr=0.01)

    trainer = Trainer(model, ce_loss, optimizer)
    trainer.fit(loader, epochs=5)
    print("MLP Classifier trained successfully!")
