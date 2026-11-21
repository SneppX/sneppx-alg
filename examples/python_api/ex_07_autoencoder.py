"""
Example 7: Autoencoder with SNEPPX-Alg API
"""
import numpy as np
from SneppX_ALG.interface_bindings.pytorch_api import Tensor, Module, Linear, Sequential, Adam, Trainer, Dataset, DataLoader

class Autoencoder(Module):
    def __init__(self):
        super().__init__()
        self.encoder = Sequential(Linear(20, 10), Linear(10, 5))
        self.decoder = Sequential(Linear(5, 10), Linear(10, 20))

    def forward(self, x):
        return self.decoder(self.encoder(x))

def mse_loss(preds, targets):
    # Autoencoder targets are the input themselves
    diff = preds.data - targets.data
    return Tensor(np.mean(diff ** 2), requires_grad=True)

if __name__ == "__main__":
    data = np.random.randn(100, 20).astype(np.float32)
    dataset = [(Tensor(d), Tensor(d)) for d in data]
    loader = DataLoader(dataset, batch_size=20, shuffle=True)
    
    model = Autoencoder()
    optimizer = Adam(model.parameters(), lr=0.01)
    
    trainer = Trainer(model, mse_loss, optimizer)
    trainer.fit(loader, epochs=5)
    print("Autoencoder trained successfully!")
