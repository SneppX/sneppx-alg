"""Keras-style layer API.

Ergonomic ``Sequential`` container with ``add`` / ``compile`` / ``fit`` /
``evaluate`` / ``predict`` / ``summary`` plus layer factory functions
(``Dense``, ``Conv2D``, ``MaxPool2D``, ``Flatten``, activations, etc.),
implemented on top of the existing ``nn`` modules and the autograd Tensor.

Typical usage::

    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(8,)))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=5, batch_size=32)
    scores = model.evaluate(x_test, y_test)
    preds = model.predict(x_test)

Note: gradient flow is currently recorded through the autograd Tensor operators
(``Dense``/matmul paths); convolutional/pooling layers execute in the forward
pass without a recorded backward.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .tensor import Tensor
from . import nn as _nn
from . import train as _train
from . import advanced_ops as _advanced_ops

__all__ = [
    "Sequential",
    "Model",
    "Dense",
    "Conv2D",
    "MaxPool2D",
    "AveragePool2D",
    "Flatten",
    "Dropout",
    "Activation",
    "BatchNormalization",
    "LayerNorm",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "GELU",
    "SiLU",
    "Softmax",
    "Input",
]

_ACTIVATIONS = {
    "relu": _nn.ReLU,
    "sigmoid": _nn.Sigmoid,
    "tanh": _nn.Tanh,
    "gelu": _nn.GELU,
    "silu": _nn.SiLU,
    "softmax": None,  # handled specially via softmax path below
    "linear": None,
}

_OPTIMIZERS = {
    "sgd": None,  # resolved to keras_api pure-python optimizers
    "adam": None,
    "adamw": None,
}

_LOSSES = {
    "mse": _train.MSELoss,
    "mean_squared_error": _train.MSELoss,
    "mae": _train.MAELoss,
    "mean_absolute_error": _train.MAELoss,
    "bce": _train.BCELoss,
    "binary_crossentropy": _train.BCELoss,
    "categorical_crossentropy": _train.CrossEntropyLoss,
    "cross_entropy": _train.CrossEntropyLoss,
    "nll": _train.NLLLoss,
    "kld": _train.KLDivLoss,
    "kl_divergence": _train.KLDivLoss,
}


def _as_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor.from_numpy(np.asarray(x, dtype=np.float32))


# ===========================================================================
#  Layer factories (modules)
# ===========================================================================


class Dense(_nn.Module):
    def __init__(
        self,
        units: int,
        activation: Optional[str] = None,
        use_bias: bool = True,
        input_shape: Optional[Tuple[int, ...]] = None,
        dtype: str = "float32",
    ):
        super().__init__()
        self.units = units
        self.activation_name = activation
        self.input_shape = tuple(input_shape) if input_shape else None
        self._dense = _nn.Linear(0, units, bias=use_bias, dtype=dtype)
        self._act = _activation_layer(activation)
        if activation == "softmax":
            self._softmax = True
        else:
            self._softmax = False

    def build(self, in_features: int):
        if self._dense.in_features == 0:
            w = Tensor.randn((self.units, in_features)) / np.sqrt(in_features)
            b = Tensor.zeros((self.units,)) if self._dense.bias is not None else None
            self._dense.in_features = in_features
            self._dense.weight = w
            if b is not None:
                self._dense.bias = b
            self._built_in = in_features

    def forward(self, x: Tensor) -> Tensor:
        if self._dense.in_features == 0:
            self.build(x.shape[-1])
        out = self._dense(x)
        if self._softmax:
            return out.softmax(dim=-1)
        if self._act is not None:
            out = self._act(out)
        return out

    def parameters(self):
        return self._dense.parameters()


class Activation(_nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._act = _activation_layer(name)
        self._softmax = name == "softmax"

    def forward(self, x: Tensor) -> Tensor:
        if self._softmax:
            return x.softmax(dim=-1)
        if self._act is None:
            return x
        return self._act(x)


def _activation_layer(name: Optional[str]):
    if not name or name == "linear":
        return None
    if name == "softmax":
        return None
    cls = _ACTIVATIONS.get(name)
    if cls is None:
        raise ValueError(f"unknown activation: {name!r}")
    return cls()


class ReLU(_nn.ReLU):
    pass


class Sigmoid(_nn.Sigmoid):
    pass


class Tanh(_nn.Tanh):
    pass


class GELU(_nn.GELU):
    pass


class SiLU(_nn.SiLU):
    pass


class Softmax(_nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(dim=self.dim)


class Dropout(_nn.Module):
    def __init__(self, rate: float = 0.5, seed: Optional[int] = None):
        super().__init__()
        self.rate = rate
        self.seed = seed
        self._drop = _nn.Dropout(rate)

    def forward(self, x: Tensor) -> Tensor:
        if self._training:
            return self._drop(x)
        return x


class LayerNorm(_nn.Module):
    def __init__(self, normalized_shape: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()
        size = int(normalized_shape) if isinstance(normalized_shape, int) else int(
            normalized_shape[0]
        )
        self._ln = _nn.LayerNorm(size, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self._ln(x)


class BatchNormalization(_nn.Module):
    def __init__(
        self, num_features: int, momentum: float = 0.9, eps: float = 1e-5,
        axis: int = -1, dtype: str = "float32",
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.axis = axis
        self.gamma = Tensor.ones((num_features,), dtype=dtype)
        self.beta = Tensor.zeros((num_features,), dtype=dtype)
        self.running_mean = Tensor.zeros((num_features,), dtype=dtype)
        self.running_var = Tensor.ones((num_features,), dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        axis = self.axis % x.ndim
        if self._training:
            axes = tuple(i for i in range(x.ndim) if i != axis)
            mean = x.mean(dim=axes)
            var = (x * x).mean(dim=axes) - mean * mean
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        shape = [1] * x.ndim
        shape[axis] = self.num_features
        mean = mean.reshape(shape)
        var = var.reshape(shape)
        gamma = self.gamma.reshape(shape)
        beta = self.beta.reshape(shape)
        return (x - mean) / (var + self.eps).sqrt() * gamma + beta


class Conv2D(_nn.Module):
    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        activation: Optional[str] = None,
        use_bias: bool = True,
        dtype: str = "float32",
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        self.strides = (
            (strides, strides) if isinstance(strides, int) else tuple(strides)
        )
        self.padding = (
            (padding, padding) if isinstance(padding, int) else tuple(padding)
        )
        self.activation_name = activation
        self._act = _activation_layer(activation)
        self._softmax = activation == "softmax"
        self.weight: Optional[Tensor] = None
        self.bias: Optional[Tensor] = None
        self.dtype = dtype

    def build(self, in_channels: int):
        kh, kw = self.kernel_size
        w = Tensor.randn((self.filters, in_channels, kh, kw), dtype=self.dtype) * 0.1
        self.weight = w
        self.bias = Tensor.zeros((self.filters,), dtype=self.dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.weight is None:
            self.build(x.shape[1] if x.ndim == 4 else x.shape[-1])
        out = _advanced_ops.conv2d(
            x, self.weight, self.bias, stride=self.strides, padding=self.padding
        )
        if self._softmax:
            return out.softmax(dim=1)
        if self._act is not None:
            out = self._act(out)
        return out


class MaxPool2D(_nn.Module):
    def __init__(self, pool_size: Union[int, Tuple[int, int]] = 2,
                 strides: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return _advanced_ops.max_pool2d(
            x, self.pool_size, stride=self.strides, padding=self.padding
        )


class AveragePool2D(_nn.Module):
    def __init__(self, pool_size: Union[int, Tuple[int, int]] = 2,
                 strides: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return _advanced_ops.avg_pool2d(
            x, self.pool_size, stride=self.strides, padding=self.padding
        )


class Flatten(_nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        n = 1
        for d in x.shape[1:]:
            n *= d
        return x.reshape((x.shape[0], n))


# ===========================================================================
#  Functional-style Input
# ===========================================================================


class _InputSpec:
    """Shape-tagged input marker for ``Sequential(input_shape=...)``."""

    def __init__(self, shape: Tuple[Optional[int], ...]):
        self.shape = tuple(shape)

    @property
    def input_shape(self):
        return self.shape


def Input(shape: Tuple[Optional[int], ...]) -> _InputSpec:
    """Declare a model input shape (Keras-style)."""
    return _InputSpec(shape)


# ===========================================================================
#  Pure-python optimizers (autograd-driven, no C bridge dependency)
# ===========================================================================


class _OptimizerBase:
    """Bare-bones optimizer reading ``param.grad`` after ``backward()``.

    Leaves opt into autograd (``requires_grad_(True)``) so ``backward()``
    accumulates into ``p.grad``.
    """

    def __init__(self, params: List[Tensor], lr: float):
        self.params = params
        self.lr = lr
        for p in params:
            p.requires_grad_(True)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad_()

    def step(self):
        raise NotImplementedError


class _SGD(_OptimizerBase):
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._vel = [np.zeros_like(p.data) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            self._vel[i] = self.momentum * self._vel[i] + g
            p.data = p.data - self.lr * self._vel[i]


class _Adam(_OptimizerBase):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.b0, self.b1 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            self.m[i] = self.b0 * self.m[i] + (1 - self.b0) * g
            self.v[i] = self.b1 * self.v[i] + (1 - self.b1) * (g * g)
            mh = self.m[i] / (1 - self.b0 ** self.t)
            vh = self.v[i] / (1 - self.b1 ** self.t)
            p.data = p.data - self.lr * mh / (np.sqrt(vh) + self.eps)


class _AdamW(_Adam):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, betas, eps, weight_decay)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            p.data = p.data - self.lr * self.weight_decay * p.data
            self.m[i] = self.b0 * self.m[i] + (1 - self.b0) * g
            self.v[i] = self.b1 * self.v[i] + (1 - self.b1) * (g * g)
            mh = self.m[i] / (1 - self.b0 ** self.t)
            vh = self.v[i] / (1 - self.b1 ** self.t)
            p.data = p.data - self.lr * mh / (np.sqrt(vh) + self.eps)


# ===========================================================================
#  Keras-style Sequential
# ===========================================================================


class Sequential:
    """A Keras-style linear stack of layers with compile/fit/evaluate/predict."""

    def __init__(
        self,
        *layers,
        input_shape: Optional[Tuple[Optional[int], ...]] = None,
        name: str = "sequential",
    ):
        self.name = name
        self.layers: List[_nn.Module] = []
        if len(layers) == 1 and isinstance(layers[0], dict):
            cfg = layers[0]
            input_dim = cfg.get("input_dim", 64)
            output_dim = cfg.get("output_dim", 64)
            self.add(_nn.Linear(input_dim, output_dim))
        else:
            for layer in layers:
                self.add(layer)
        self._input_shape: Optional[Tuple[Optional[int], ...]] = tuple(
            input_shape
        ) if input_shape else None
        self._optimizer: Optional[_train.Optimizer] = None
        self._loss_fn: Optional[Callable] = None
        self._metrics: List[Tuple[str, Callable]] = []
        self.history: Dict[str, List[float]] = {}
        self._built = False

    # ---- construction ----------------------------------------------------
    def add(self, layer: _nn.Module):
        if isinstance(layer, _nn.Module) or hasattr(layer, "forward"):
            self.layers.append(layer)
        else:
            raise TypeError(f"layer must be a Module, got {type(layer).__name__}")

    @property
    def input_shape(self) -> Optional[Tuple[Optional[int], ...]]:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value):
        self._input_shape = tuple(value) if value else None

    # ---- compile ---------------------------------------------------------
    def compile(
        self,
        optimizer: Union[str, _train.Optimizer] = "adam",
        loss: Union[str, Callable] = "mse",
        metrics: Optional[Sequence[str]] = None,
        lr: float = 0.001,
    ):
        if not self.layers:
            raise ValueError("compile() called before adding any layers")
        self._build_graph()
        if isinstance(optimizer, str):
            name = optimizer.lower()
            if name == "sgd":
                self._optimizer = _SGD(self.parameters(), lr=lr)
            elif name == "adam":
                self._optimizer = _Adam(self.parameters(), lr=lr)
            elif name == "adamw":
                self._optimizer = _AdamW(self.parameters(), lr=lr)
            else:
                raise ValueError(f"unknown optimizer: {optimizer!r}")
        else:
            self._optimizer = optimizer
            for p in self.parameters():
                p.requires_grad_(True)
        if isinstance(loss, str):
            cls = _LOSSES.get(loss.lower())
            if cls is None:
                raise ValueError(f"unknown loss: {loss!r}")
            self._loss_fn = cls()
        else:
            self._loss_fn = loss
        self._metrics = []
        for m in metrics or []:
            if isinstance(m, str):
                self._metrics.append((m, _metric_fn(m)))
            elif callable(m):
                self._metrics.append((getattr(m, "__name__", "metric"), m))
            else:
                raise TypeError(f"bad metric: {m!r}")
        return self

    # ---- weights ---------------------------------------------------------
    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in self.layers:
            for p in getattr(layer, "parameters", lambda: [])():
                params.append(p)
        return params

    def _build_graph(self):
        """Trigger lazy layer builds via a dry-run so weights exist before
        the optimizer captures them."""
        if self._input_shape is None:
            return
        try:
            x = Tensor.zeros((1,) + tuple(
                int(d) if d is not None else 1 for d in self._input_shape
            ))
            for layer in self.layers:
                x = layer(x)
        except (TypeError, ValueError):
            return

    def count_params(self) -> int:
        total = 0
        for p in self.parameters():
            n = 1
            for d in p.shape:
                n *= d
            total += n
        return total

    def get_weights(self) -> List[np.ndarray]:
        return [p.numpy().copy() for p in self.parameters()]

    def set_weights(self, weights: Sequence[np.ndarray]):
        params = self.parameters()
        if len(weights) != len(params):
            raise ValueError(
                f"expected {len(params)} weight arrays, got {len(weights)}"
            )
        for p, w in zip(params, weights):
            p.data = np.asarray(w, dtype=p.data.dtype)

    def save_weights(self, path: str):
        params = self.parameters()
        payload = {
            "shapes": [list(p.shape) for p in params],
            "weights": [p.numpy().tolist() for p in params],
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    def load_weights(self, path: str):
        with open(path) as f:
            payload = json.load(f)
        params = self.parameters()
        if len(payload["weights"]) != len(params):
            raise ValueError(
                f"file has {len(payload['weights'])} arrays, model has {len(params)}"
            )
        for p, w in zip(params, payload["weights"]):
            p.data = np.asarray(w, dtype=p.data.dtype)

    # ---- forward ---------------------------------------------------------
    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = _as_tensor(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        return self(x)

    # ---- predict ---------------------------------------------------------
    def predict(self, x, batch_size: Optional[int] = None) -> np.ndarray:
        xt = _as_tensor(x)
        self._set_input_shape(xt.shape)
        return self(xt).numpy()

    def train(self):
        self._training = True
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()

    def eval(self):
        self._training = False
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()

    def _set_input_shape(self, shape):
        if self._input_shape is None and shape:
            self._input_shape = tuple(int(s) for s in shape)

    # ---- fit -------------------------------------------------------------
    def fit(
        self,
        x,
        y=None,
        epochs: int = 1,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        verbose: int = 1,
        validation_data: Optional[Tuple] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, List[float]]:
        if self._optimizer is None or self._loss_fn is None:
            raise RuntimeError("call compile() before fit()")
        x_np = x.numpy() if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
        y_np = None
        if y is not None:
            y_np = y.numpy() if isinstance(y, Tensor) else np.asarray(y, dtype=np.float32)
        self._set_input_shape(x_np.shape)

        callbacks = list(callbacks or [])
        self.history = {}
        n = x_np.shape[0]

        for epoch in range(epochs):
            order = np.arange(n)
            if shuffle:
                np.random.shuffle(order)
            bs = batch_size or n
            epoch_losses: List[float] = []
            epoch_metrics: Dict[str, List[float]] = {k: [] for k, _ in self._metrics}
            for start in range(0, n, bs):
                idx = order[start : start + bs]
                xb = _as_tensor(x_np[idx])
                self._optimizer.zero_grad()
                pred = self(xb)
                if y_np is not None:
                    loss = self._loss_fn(pred, _as_tensor(y_np[idx]))
                else:
                    loss = self._loss_fn(pred)
                loss.backward()
                self._optimizer.step()
                epoch_losses.append(float(loss.item()))
                if y_np is not None:
                    pred_np = pred.numpy()
                    for name, fn in self._metrics:
                        epoch_metrics[name].append(
                            float(fn(pred_np, y_np[idx]))
                        )
            self.history.setdefault("loss", []).append(
                float(np.mean(epoch_losses))
            )
            for name, _ in self._metrics:
                self.history.setdefault(name, []).append(
                    float(np.mean(epoch_metrics[name]))
                )
            line = f"Epoch {epoch + 1}/{epochs} - loss: {self.history['loss'][-1]:.4f}"
            for name, _ in self._metrics:
                line += f" - {name}: {self.history[name][-1]:.4f}"
            if validation_data is not None:
                val = self.evaluate(*validation_data, verbose=0)
                self.history.setdefault("val_loss", []).append(float(val[0]))
                for i, (name, _) in enumerate(self._metrics):
                    self.history.setdefault(f"val_{name}", []).append(float(val[i + 1]))
                line += f" - val_loss: {self.history['val_loss'][-1]:.4f}"
            if verbose:
                print(line)
            for cb in callbacks:
                cb(self)
        return self.history

    # ---- evaluate --------------------------------------------------------
    def evaluate(
        self, x, y=None, batch_size: Optional[int] = None, verbose: int = 0
    ) -> List[float]:
        if self._loss_fn is None:
            raise RuntimeError("call compile() before evaluate()")
        x_np = x.numpy() if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
        y_np = None
        if y is not None:
            y_np = y.numpy() if isinstance(y, Tensor) else np.asarray(y, dtype=np.float32)
        self._set_input_shape(x_np.shape)
        pred = self(_as_tensor(x_np)).numpy()
        results: List[float] = []
        if y_np is not None:
            loss = self._loss_fn(_as_tensor(pred), _as_tensor(y_np))
            results.append(float(loss.item()))
        for name, fn in self._metrics:
            results.append(float(fn(pred, y_np) if y_np is not None else 0.0))
        return results

    # ---- summary ---------------------------------------------------------
    def summary(self) -> str:
        lines = []
        lines.append(f"Model: {self.name}")
        lines.append("-" * 60)
        lines.append(f"{'Layer':<28}{'Output shape':<20}{'Params':>10}")
        lines.append("=" * 60)
        x: Optional[Tensor] = None
        if self._input_shape:
            try:
                x = Tensor.zeros((1,) + tuple(
                    int(d) if d is not None else 1 for d in self._input_shape
                ))
            except (TypeError, ValueError):
                x = None
        total = 0
        for i, layer in enumerate(self.layers):
            if x is not None:
                try:
                    x = layer(x)
                    out = tuple(str(s) for s in x.shape)
                except Exception:
                    out = ("?",)
            else:
                out = ("?",)
            pc = 0
            for p in getattr(layer, "parameters", lambda: [])():
                n = 1
                for d in p.shape:
                    n *= d
                pc += n
            total += pc
            lines.append(
                f"{type(layer).__name__:<28}{', '.join(out):<20}{pc:>10}"
            )
        lines.append("=" * 60)
        lines.append(f"Total params: {total}")
        return "\n".join(lines)

    # ---- state -----------------------------------------------------------
    def state_dict(self) -> Dict[str, np.ndarray]:
        state: Dict[str, np.ndarray] = {}
        for layer in self.layers:
            for name, p in getattr(layer, "named_parameters", lambda: [])():
                state[name] = p.numpy().copy()
        return state


Model = Sequential


def _metric_fn(name: str) -> Callable:
    name = name.lower()
    if name in ("acc", "accuracy"):
        return _accuracy
    if name in ("mse", "mean_squared_error"):
        return _mse
    if name in ("mae", "mean_absolute_error"):
        return _mae
    raise ValueError(f"unknown metric: {name!r}")


def _accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    pred_arg = np.argmax(pred, axis=-1)
    t = target if target.ndim == pred.ndim else target
    if t.ndim == pred.ndim and t.shape[-1] == pred.shape[-1]:
        t = np.argmax(t, axis=-1)
    t = t.reshape(pred_arg.shape)
    return float(np.mean(pred_arg == t))


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))
