"""Training API — Trainer, data pipeline, metrics."""

from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union
from .core import Tensor
from .nn import Module
from .optim import Optimizer, AdamW, CosineAnnealingLR
import numpy as np
import time


class DataLoader:
    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __len__(self):
        return max(1, len(self.dataset) // self.batch_size)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self._pos = 0
        return self

    def __next__(self):
        if self._pos >= len(self.dataset):
            raise StopIteration
        batch_idx = self.indices[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        batch = [self.dataset[i] for i in batch_idx]
        xs = Tensor(np.stack([b[0] for b in batch]))
        ys = Tensor(np.stack([b[1] for b in batch]))
        return xs, ys


class Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.accuracies = []
        self.times = []

    def update(self, loss: float, accuracy: float = 0.0):
        self.losses.append(loss)
        self.accuracies.append(accuracy)

    def avg_loss(self):
        return float(np.mean(self.losses[-100:])) if self.losses else 0.0

    def avg_accuracy(self):
        return float(np.mean(self.accuracies[-100:])) if self.accuracies else 0.0


class Trainer:
    def __init__(self, model: Module, optimizer: Optimizer, loss_fn: Callable,
                 device: str = "cpu", gradient_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.metrics = Metrics()
        self.scheduler = None
        self._step = 0

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _compute_loss(self, batch) -> Tensor:
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        return loss

    def _accuracy(self, pred: Tensor, target: Tensor) -> float:
        p = pred.numpy()
        t = target.numpy()
        if t.ndim > 1 and t.shape[-1] > 1:
            p = p.argmax(axis=-1)
            t = t.argmax(axis=-1)
        return float((p == t).mean())

    def train_epoch(self, train_loader: DataLoader) -> Dict:
        self.model.train()
        self.metrics.reset()
        epoch_start = time.time()
        for step, batch in enumerate(train_loader):
            loss = self._compute_loss(batch)
            loss_val = float(loss.numpy())
            acc = self._accuracy(*batch)
            loss.backward()
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    self._clip_gradients()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()
            self.metrics.update(loss_val, acc)
            self._step += 1
        epoch_time = time.time() - epoch_start
        return {
            "loss": self.metrics.avg_loss(),
            "accuracy": self.metrics.avg_accuracy(),
            "time": epoch_time,
        }

    def evaluate(self, eval_loader: DataLoader) -> Dict:
        self.model.eval()
        losses, accs = [], []
        for batch in eval_loader:
            loss = self._compute_loss(batch)
            acc = self._accuracy(*batch)
            losses.append(float(loss.numpy()))
            accs.append(acc)
        return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(accs))}

    def _clip_gradients(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                g = p.grad.numpy()
                total_norm += np.sum(g ** 2)
        total_norm = np.sqrt(total_norm)
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / total_norm
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad = Tensor(p.grad.numpy() * scale, dtype=p.grad.dtype)


class DataPipeline:
    def __init__(self):
        self.transforms = []

    def add_transform(self, fn: Callable):
        self.transforms.append(fn)

    def __call__(self, x):
        for fn in self.transforms:
            x = fn(x)
        return x


class Normalize:
    def __init__(self, mean: Union[float, List], std: Union[float, List]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, x):
        return (x - self.mean) / self.std


class RandomFlip:
    def __call__(self, x):
        if np.random.rand() > 0.5:
            return np.fliplr(x) if x.ndim >= 2 else x
        return x


class ToTensor:
    def __call__(self, x):
        if isinstance(x, Tensor):
            return x
        return Tensor(x, dtype="float32")