"""
Example 5: NLP Sentiment Analysis with Embedding Layer using SNEPPX-Alg API
"""
import numpy as np
from SneppX_ALG.interface_bindings.pytorch_api import Tensor, Module, Embedding, Linear, Adam, Trainer, Dataset, DataLoader

class SentimentDataset(Dataset):
    def __init__(self, n=150, seq_len=8):
        self.x = np.random.randint(0, 100, size=(n, seq_len)).astype(np.int64)
        self.y = np.random.randint(0, 2, size=(n,)).astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return Tensor(self.x[idx]), self.y[idx]

class SentimentModel(Module):
    def __init__(self):
        super().__init__()
        self.embed = Embedding(num_embeddings=100, embedding_dim=16)
        self.fc = Linear(16, 2)

    def forward(self, x):
        emb = self.embed(x)
        # Average pooling over sequence length
        pooled = Tensor(np.mean(emb.data, axis=1), requires_grad=emb.requires_grad)
        return self.fc(pooled)

def loss_fn(preds, targets):
    exps = np.exp(preds.data - np.max(preds.data, axis=-1, keepdims=True))
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    n = len(targets)
    targets_arr = np.array(targets)
    loss = -np.mean(np.log(probs[np.arange(n), targets_arr] + 1e-7))
    return Tensor(loss, requires_grad=True)

if __name__ == "__main__":
    dataset = SentimentDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SentimentModel()
    optimizer = Adam(model.parameters(), lr=0.01)

    trainer = Trainer(model, loss_fn, optimizer)
    trainer.fit(loader, epochs=3)
    print("Sentiment NLP model trained successfully!")
