"""
SNEPPX-Alg PyTorch-Like High-Level Python API Layer.
Provides a familiar PyTorch-compatible API (Tensor, nn.Module, optim, data.DataLoader, metrics, trainer)
backed by SNEPPX-Alg's C/C++ cognitive processing engine and autograd tape.
"""

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Callable, Dict, Any, Union
import numpy as np

try:
    from . import _arix_c as _backend
except ImportError:
    try:
        from . import _SNEPPX_c as _backend
    except ImportError:
        _backend = None


# ============================================================================
# 1. TENSOR & AUTOGRAD
# ============================================================================

class Tensor:
    """
    High-level Tensor wrapping NumPy arrays and SNEPPX-Alg C backend buffers
    with PyTorch-like automatic differentiation tape tracing.
    """
    def __init__(self, data: Union[np.ndarray, list, float, int], requires_grad: bool = False, dtype=np.float32):
        if isinstance(data, Tensor):
            self.data = np.array(data.data, dtype=dtype)
            self.requires_grad = requires_grad or data.requires_grad
            self._grad = np.zeros_like(self.data) if self.requires_grad else None
            self._grad_fn = None
            self._creator = data
        else:
            self.data = np.array(data, dtype=dtype)
            self.requires_grad = requires_grad
            self._grad = np.zeros_like(self.data) if self.requires_grad else None
            self._grad_fn = None
            self._creator = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def grad(self) -> Optional['Tensor']:
        if self._grad is None:
            return None
        t = Tensor(self._grad, requires_grad=False)
        return t

    @grad.setter
    def grad(self, value):
        if value is None:
            self._grad = None
        elif isinstance(value, Tensor):
            self._grad = value.data.copy()
        else:
            self._grad = np.array(value, dtype=self.data.dtype)

    def backward(self, gradient: Optional['Tensor'] = None):
        """Backpropagate gradients through the computation graph."""
        if not self.requires_grad:
            return
        if gradient is None:
            if self.shape == () or self.data.size == 1:
                grad_arr = np.ones_like(self.data)
            else:
                raise RuntimeError("Grad can be implicitly created only for scalar outputs")
        else:
            grad_arr = gradient.data

        if self._grad is None:
            self._grad = grad_arr.copy()
        else:
            self._grad += grad_arr

        if self._grad_fn is not None:
            self._grad_fn(self._grad)

    def zero_grad(self):
        """Reset gradients to zero."""
        if self._grad is not None:
            self._grad.fill(0.0)

    def item(self) -> float:
        """Extract Python float from scalar tensor."""
        return float(self.data.item())

    def numpy(self) -> np.ndarray:
        """Convert tensor to underlying NumPy array."""
        return self.data

    def to_c_backend(self):
        """Seamless conversion to C backend tensor buffer if available."""
        if _backend and hasattr(_backend, "Tensor"):
            return _backend.Tensor(self.data)
        return self.data

    @classmethod
    def from_c_backend(cls, c_tensor, requires_grad: bool = False) -> 'Tensor':
        """Create high-level Tensor from C backend tensor."""
        if hasattr(c_tensor, "numpy"):
            arr = c_tensor.numpy()
        else:
            arr = np.array(c_tensor)
        return cls(arr, requires_grad=requires_grad)

    def __add__(self, other):
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other_t.data, requires_grad=self.requires_grad or other_t.requires_grad)
        def grad_fn(grad):
            if self.requires_grad:
                self.backward(Tensor(grad))
            if other_t.requires_grad:
                other_t.backward(Tensor(grad))
        out._grad_fn = grad_fn
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other_t.data, requires_grad=self.requires_grad or other_t.requires_grad)
        def grad_fn(grad):
            if self.requires_grad:
                self.backward(Tensor(grad))
            if other_t.requires_grad:
                other_t.backward(Tensor(-grad.data if isinstance(grad, Tensor) else -grad))
        out._grad_fn = grad_fn
        return out

    def __mul__(self, other):
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other_t.data, requires_grad=self.requires_grad or other_t.requires_grad)
        def grad_fn(grad):
            if self.requires_grad:
                self.backward(Tensor(grad.data * other_t.data))
            if other_t.requires_grad:
                other_t.backward(Tensor(grad.data * self.data))
        out._grad_fn = grad_fn
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other_t.data, requires_grad=self.requires_grad or other_t.requires_grad)
        def grad_fn(grad):
            grad_arr = grad.data if isinstance(grad, Tensor) else grad
            if self.requires_grad:
                self.backward(Tensor(grad_arr @ other_t.data.T))
            if other_t.requires_grad:
                other_t.backward(Tensor(self.data.T @ grad_arr))
        out._grad_fn = grad_fn
        return out

    def __repr__(self):
        return f"Tensor({self.data.tolist()}, requires_grad={self.requires_grad})"


# ============================================================================
# 2. NEURAL NETWORK MODULES (sneppx.nn)
# ============================================================================

class Parameter(Tensor):
    """A Tensor that is considered a trainable module parameter."""
    def __init__(self, data: Union[np.ndarray, list]):
        super().__init__(data, requires_grad=True)


class Module:
    """Base class for all neural network modules mirroring PyTorch nn.Module."""
    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, 'Module'] = {}
        self.training = True

    def register_parameter(self, name: str, param: Optional[Parameter]):
        if param is None:
            self._parameters.pop(name, None)
        else:
            self._parameters[name] = param

    def add_module(self, name: str, module: Optional['Module']):
        if module is None:
            self._modules.pop(name, None)
        else:
            self._modules[name] = module

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        super().__setattr__(name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self) -> List[Parameter]:
        params = list(self._parameters.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def train(self, mode: bool = True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    """Fully connected linear layer: y = xW^T + b."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        self.weight = Parameter(np.random.uniform(-bound, bound, (out_features, in_features)))
        self.bias = Parameter(np.random.uniform(-bound, bound, (out_features,))) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ Tensor(self.weight.data.T, requires_grad=self.weight.requires_grad)
        if self.bias is not None:
            out = out + Tensor(self.bias.data, requires_grad=self.bias.requires_grad)
        return out


class LayerNorm(Module):
    """Layer Normalization."""
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Parameter(np.ones(normalized_shape))
        self.bias = Parameter(np.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        norm_data = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = norm_data * self.weight.data + self.bias.data
        return Tensor(out_data, requires_grad=x.requires_grad)


class Dropout(Module):
    """Dropout regularization layer."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0:
            mask = (np.random.rand(*x.shape) >= self.p) / (1.0 - self.p)
            return Tensor(x.data * mask, requires_grad=x.requires_grad)
        return x


class Embedding(Module):
    """Lookup table for embeddings."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = Parameter(np.random.normal(0, 1.0 / math.sqrt(embedding_dim), (num_embeddings, embedding_dim)))

    def forward(self, indices: Union[list, np.ndarray, Tensor]) -> Tensor:
        idx_arr = indices.numpy() if isinstance(indices, Tensor) else np.array(indices)
        out_data = self.weight.data[idx_arr]
        return Tensor(out_data, requires_grad=self.weight.requires_grad)


class Conv2d(Module):
    """2D Convolutional Layer."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        bound = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = Parameter(np.random.uniform(-bound, bound, (out_channels, in_channels, kernel_size, kernel_size)))
        self.bias = Parameter(np.zeros((out_channels,)))

    def forward(self, x: Tensor) -> Tensor:
        # Simplified batch convolution forward pass
        batch_size, in_c, in_h, in_w = x.data.shape
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        padded = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        out = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=np.float32)
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        hs = oh * self.stride
                        ws = ow * self.stride
                        patch = padded[b, :, hs:hs+self.kernel_size, ws:ws+self.kernel_size]
                        out[b, oc, oh, ow] = np.sum(patch * self.weight.data[oc]) + self.bias.data[oc]
        return Tensor(out, requires_grad=x.requires_grad)


class Sequential(Module):
    """Sequential container of modules."""
    def __init__(self, *args: Module):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x


# ============================================================================
# 3. OPTIMIZERS (sneppx.optim)
# ============================================================================

class Optimizer:
    """Base class for optimizers."""
    def __init__(self, params: List[Parameter], lr: float = 1e-3):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum."""
    def __init__(self, params: List[Parameter], lr: float = 1e-2, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.data
            if self.momentum != 0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                update = self.velocities[i]
            else:
                update = grad
            p.data -= self.lr * update


class Adam(Optimizer):
    """Adam Optimizer."""
    def __init__(self, params: List[Parameter], lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """AdamW Optimizer with decoupled weight decay."""
    def __init__(self, params: List[Parameter], lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(params, lr, betas, eps)
        self.weight_decay = weight_decay

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.weight_decay != 0:
                p.data -= self.lr * self.weight_decay * p.data
            grad = p.grad.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================================
# 4. DATA LOADER (sneppx.data)
# ============================================================================

class Dataset:
    """Base Dataset class."""
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


class TensorDataset(Dataset):
    """Dataset that indexes one or more arrays along their first dimension."""

    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor")
        arrays = tuple(np.asarray(tensor) for tensor in tensors)
        if any(array.ndim == 0 for array in arrays):
            raise ValueError("TensorDataset inputs must have a batch dimension")
        length = arrays[0].shape[0]
        if any(array.shape[0] != length for array in arrays[1:]):
            raise ValueError("all TensorDataset inputs must have the same first dimension")
        self.tensors = arrays
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        return tuple(tensor[index] for tensor in self.tensors)


class Subset(Dataset):
    """View a Dataset through a fixed sequence of indices."""

    def __init__(self, dataset: Dataset, indices):
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


class ConcatDataset(Dataset):
    """Concatenate multiple datasets into one indexable dataset."""

    def __init__(self, datasets):
        datasets = tuple(datasets)
        if not datasets:
            raise ValueError("ConcatDataset requires at least one dataset")
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int):
        from bisect import bisect_right
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("ConcatDataset index out of range")
        dataset_index = bisect_right(self.cumulative_sizes, index)
        previous_end = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        return self.datasets[dataset_index][index - previous_end]


class DataLoader:
    """Batch a Dataset with deterministic optional shuffling and threaded loading.

    num_workers uses a thread pool (not subprocesses) to fetch samples.
    Supply collate_fn for custom or variable-length batch collation.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        if isinstance(batch_size, (bool, np.bool_)):
            raise ValueError("batch_size must be a positive integer")
        try:
            normalized_batch_size = int(batch_size)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("batch_size must be a positive integer") from exc
        if normalized_batch_size <= 0 or normalized_batch_size != batch_size:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(num_workers, (bool, np.bool_)):
            raise ValueError("num_workers must be a non-negative integer")
        try:
            normalized_workers = int(num_workers)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("num_workers must be a non-negative integer") from exc
        if normalized_workers < 0 or normalized_workers != num_workers:
            raise ValueError("num_workers must be a non-negative integer")
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise TypeError("dataset must implement __len__ and __getitem__")

        self.dataset = dataset
        self.batch_size = normalized_batch_size
        self.shuffle = bool(shuffle)
        self.num_workers = normalized_workers
        self.drop_last = bool(drop_last)
        self.collate_fn = collate_fn
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(None if self.seed is None else self.seed + self._epoch)
            rng.shuffle(indices)
        self._epoch += 1

        if self.drop_last:
            indices = indices[: (len(indices) // self.batch_size) * self.batch_size]

        chunks = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]

        if self.num_workers > 0:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for chunk in chunks:
                    samples = list(executor.map(self.dataset.__getitem__, chunk))
                    yield self._collate(samples)
        else:
            for chunk in chunks:
                yield self._collate([self.dataset[index] for index in chunk])

    def _collate(self, batch):
        if self.collate_fn is not None:
            return self.collate_fn(batch)
        if not batch:
            return batch
        elem = batch[0]
        if isinstance(elem, tuple):
            if not all(isinstance(sample, tuple) and len(sample) == len(elem) for sample in batch):
                raise ValueError("all tuple samples must have the same structure")
            zipped = list(zip(*batch))
            return tuple(Tensor(np.array(values)) for values in zipped)
        return Tensor(np.array(batch))

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


# ============================================================================
# 5. METRICS (sneppx.metrics)
# ============================================================================

def accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute classification accuracy."""
    pred_labels = np.argmax(preds, axis=-1) if preds.ndim > 1 else np.round(preds)
    return float(np.mean(pred_labels == targets))


def precision_recall_f1(preds: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float]:
    """Compute binary Precision, Recall, and F1 Score."""
    pred_labels = (preds >= 0.5).astype(int) if preds.ndim == 1 else np.argmax(preds, axis=-1)
    tp = np.sum((pred_labels == 1) & (targets == 1))
    fp = np.sum((pred_labels == 1) & (targets == 0))
    fn = np.sum((pred_labels == 0) & (targets == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


# ============================================================================
# 6. TRAINER (sneppx.trainer)
# ============================================================================

class Trainer:
    """High-level training loop controller with validation, checkpointing, and early stopping."""
    def __init__(self, model: Module, criterion: Callable, optimizer: Optimizer, val_loader: Optional[DataLoader] = None, patience: int = 3):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def fit(self, train_loader: DataLoader, epochs: int):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}", end="")

            if self.val_loader:
                val_loss = self.evaluate(self.val_loader)
                print(f" | Val Loss: {val_loss:.4f}", end="")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.patience:
                        print("\nEarly stopping triggered.")
                        break
            print()

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            preds = self.model(batch_x)
            loss = self.criterion(preds, batch_y)
            total_loss += loss.item()
        return total_loss / len(loader)
