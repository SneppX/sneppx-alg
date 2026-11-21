"""
Example 4: CNN Image Classification using SNEPPX-Alg API
"""
import numpy as np
from SneppX_ALG.interface_bindings.pytorch_api import Tensor, Module, Conv2d, Linear, Sequential, Adam, Trainer, Dataset, DataLoader

class TinyImageDataset(Dataset):
    def __init__(self, n=100):
        # Batch of 100 images: 1 channel, 14x14
        self.x = np.random.randn(n, 1, 14, 14).astype(np.float32)
        self.y = np.random.randint(0, 2, size=(n,)).astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return Tensor(self.x[idx]), self.y[idx]

class SimpleCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.fc = Linear(4 * 14 * 14, 2)

    def forward(self, x):
        h = self.conv(x)
        # Flatten
        b, c, h_dim, w_dim = h.shape
        flat = Tensor(h.data.reshape(b, -1), requires_grad=h.requires_grad)
        return self.fc(flat)

def loss_fn(preds, targets):
    exps = np.exp(preds.data - np.max(preds.data, axis=-1, keepdims=True))
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    n = len(targets)
    targets_arr = np.array(targets)
    loss = -np.mean(np.log(probs[np.arange(n), targets_arr] + 1e-7))
    return Tensor(loss, requires_grad=True)

if __name__ == "__main__":
    dataset = TinyImageDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SimpleCNN()
    optimizer = Adam(model.parameters(), lr=0.01)

    trainer = Trainer(model, loss_fn, optimizer)
    trainer.fit(loader, epochs=3)
    print("CNN Image Classifier trained successfully!")
