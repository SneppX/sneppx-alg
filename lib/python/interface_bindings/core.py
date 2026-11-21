"""Tensor Module — CPU/CUDA tensor with PyTorch-like API."""

from typing import List, Optional, Tuple, Union, Callable
import ctypes
import math
import os
import numpy as np

DTYPE_MAP = {
    "float32": (ctypes.c_float, 4),
    "float16": (ctypes.c_uint16, 2),
    "int32": (ctypes.c_int32, 4),
    "int64": (ctypes.c_int64, 8),
    "uint8": (ctypes.c_uint8, 1),
}

class _CTensorData:
    def __init__(self, shape, dtype="float32"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.size = math.prod(shape)
        self.nbytes = self.size * DTYPE_MAP[dtype][1]
        self._cptr = None
        self._is_cuda = False
        if self.size > 0:
            self._data = bytearray(self.nbytes)
    
    def _copy_from(self, arr: np.ndarray):
        if self.size > 0:
            self._data[:] = arr.tobytes()
    
    def to_numpy(self):
        arr = np.frombuffer(self._data, dtype=DTYPE_MAP[self.dtype][0]).reshape(self.shape)
        return arr.copy()

class Tensor:
    def __init__(self, data=None, shape=None, dtype="float32", device="cpu", requires_grad=False):
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        if isinstance(data, (int, float)):
            shape = shape or (1,)
            self._data = _CTensorData(shape, dtype)
            arr = np.full(shape, data, dtype=DTYPE_MAP[dtype][0])
            self._data._copy_from(arr)
        elif isinstance(data, np.ndarray):
            self._data = _CTensorData(data.shape, dtype)
            self._data._copy_from(data.astype(DTYPE_MAP[dtype][0]))
            self.shape = tuple(data.shape)
        elif isinstance(data, Tensor):
            self._data = data._data
            self.shape = data.shape
        else:
            shape = shape or (1,)
            self._data = _CTensorData(shape, dtype)
            self.shape = shape
    
    @property
    def shape(self):
        return self._data.shape
    
    @shape.setter
    def shape(self, s):
        self._data.shape = tuple(s)
    
    @property
    def ndim(self):
        return len(self.shape)
    
    @property
    def numel(self):
        return self._data.size
    
    def numpy(self) -> np.ndarray:
        return self._data.to_numpy()
    
    def to(self, device: str):
        self.device = device
        return self
    
    def cuda(self):
        return self.to("cuda")
    
    def cpu(self):
        return self.to("cpu")
    
    def item(self) -> float:
        return float(self.numpy().flat[0])
    
    def view(self, *shape):
        t = Tensor(self)
        t.shape = shape
        return t
    
    def reshape(self, *shape):
        return self.view(*shape)
    
    def clone(self):
        t = Tensor(self)
        t._data = _CTensorData(self.shape, self.dtype)
        t._data._data[:] = self._data._data[:]
        return t
    
    def copy_(self, src: "Tensor"):
        if self.shape == src.shape:
            self._data._data[:] = src._data._data[:]
        return self
    
    def zero_(self):
        self._data._data = bytearray(self._data.nbytes)
        return self
    
    def _apply_unary(self, fn: Callable):
        arr = self.numpy()
        return Tensor(fn(arr), dtype=self.dtype)
    
    def _apply_binary(self, other, fn: Callable):
        a = self.numpy()
        b = other.numpy() if isinstance(other, Tensor) else other
        return Tensor(fn(a, b), dtype=self.dtype)
    
    def __add__(self, other):
        return self._apply_binary(other, lambda a, b: a + b)
    
    def __sub__(self, other):
        return self._apply_binary(other, lambda a, b: a - b)
    
    def __mul__(self, other):
        return self._apply_binary(other, lambda a, b: a * b)
    
    def __truediv__(self, other):
        return self._apply_binary(other, lambda a, b: a / b)
    
    def __matmul__(self, other):
        a, b = self.numpy(), other.numpy()
        return Tensor(a @ b, dtype=self.dtype)
    
    def __neg__(self):
        return self._apply_unary(lambda a: -a)
    
    def __getitem__(self, key):
        arr = self.numpy()
        return Tensor(arr[key], dtype=self.dtype)
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"
    
    def __len__(self):
        return self.shape[0] if self.shape else 0
    
    def mean(self):
        return Tensor(float(self.numpy().mean()), dtype=self.dtype)
    
    def sum(self):
        return Tensor(float(self.numpy().sum()), dtype=self.dtype)
    
    def sqrt(self):
        return self._apply_unary(lambda a: np.sqrt(a))
    
    def exp(self):
        return self._apply_unary(lambda a: np.exp(a))
    
    def log(self):
        return self._apply_unary(lambda a: np.log(a + 1e-10))
    
    def abs(self):
        return self._apply_unary(lambda a: np.abs(a))
    
    def relu(self):
        return self._apply_unary(lambda a: np.maximum(0, a))
    
    def sigmoid(self):
        return self._apply_unary(lambda a: 1.0 / (1.0 + np.exp(-a)))
    
    def tanh(self):
        return self._apply_unary(lambda a: np.tanh(a))
    
    def gelu(self):
        return self._apply_unary(lambda a: 0.5 * a * (1.0 + np.tanh(0.79788456 * (a + 0.044715 * a**3))))
    
    def silu(self):
        return self._apply_unary(lambda a: a * (1.0 / (1.0 + np.exp(-a))))
    
    def softmax(self, dim=-1):
        a = self.numpy()
        e = np.exp(a - a.max(axis=dim, keepdims=True))
        return Tensor(e / e.sum(axis=dim, keepdims=True), dtype=self.dtype)
    
    def backward(self, grad_output=None):
        if not self.requires_grad:
            return
        if grad_output is None:
            grad_output = Tensor(np.ones_like(self.numpy()), dtype=self.dtype)
        self.grad = grad_output
    
    def detach(self):
        t = Tensor(self)
        t.requires_grad = False
        return t

    # Factory methods
    @staticmethod
    def zeros(shape, dtype="float32", device="cpu"):
        return Tensor(np.zeros(shape, dtype=DTYPE_MAP[dtype][0]), dtype=dtype, device=device)
    
    @staticmethod
    def ones(shape, dtype="float32", device="cpu"):
        return Tensor(np.ones(shape, dtype=DTYPE_MAP[dtype][0]), dtype=dtype, device=device)
    
    @staticmethod
    def randn(shape, dtype="float32", device="cpu"):
        return Tensor(np.random.randn(*shape).astype(DTYPE_MAP[dtype][0]), dtype=dtype, device=device)
    
    @staticmethod
    def rand(shape, dtype="float32", device="cpu"):
        return Tensor(np.random.rand(*shape).astype(DTYPE_MAP[dtype][0]), dtype=dtype, device=device)
    
    @staticmethod
    def arange(start, stop=None, step=1, dtype="float32"):
        return Tensor(np.arange(start, stop, step).astype(DTYPE_MAP[dtype][0]), dtype=dtype)
    
    @staticmethod
    def eye(n, dtype="float32"):
        return Tensor(np.eye(n).astype(DTYPE_MAP[dtype][0]), dtype=dtype)
    
    @staticmethod
    def full(shape, fill_value, dtype="float32"):
        return Tensor(np.full(shape, fill_value, dtype=DTYPE_MAP[dtype][0]), dtype=dtype)
    
    @staticmethod
    def from_numpy(arr: np.ndarray, dtype=None):
        dt = dtype or str(arr.dtype)
        return Tensor(arr, dtype=dt)


def cat(tensors: List[Tensor], dim=0) -> Tensor:
    return Tensor(np.concatenate([t.numpy() for t in tensors], axis=dim))

def stack(tensors: List[Tensor], dim=0) -> Tensor:
    return Tensor(np.stack([t.numpy() for t in tensors], axis=dim))