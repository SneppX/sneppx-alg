"""Tensor Module — Hybrid CPU/CUDA tensor with optional C backend and NumPy fallback."""

from typing import List, Optional, Tuple, Union, Callable
import ctypes
import math
import os
import numpy as np

# ---- Try to load the C (pybind11) backend ----
_C_BACKEND = None
try:
    from .. import _arix_c as _C_BACKEND
except ImportError:
    try:
        from .. import _SNEPPX_c as _C_BACKEND
    except ImportError:
        pass

if _C_BACKEND is not None:
    try:
        Dtype = _C_BACKEND.SNEPPXDtype
        Device = _C_BACKEND.SNEPPXDevice
        Layout = _C_BACKEND.SNEPPXLayout
        _HAS_C_BACKEND = True
    except AttributeError:
        _C_BACKEND = None
        _HAS_C_BACKEND = False
else:
    _HAS_C_BACKEND = False

if not _HAS_C_BACKEND:

    class Dtype:
        FLOAT32 = 0
        FLOAT64 = 1
        FLOAT16 = 2
        BFLOAT16 = 3
        INT32 = 4
        INT64 = 5
        INT16 = 6
        INT8 = 7
        UINT8 = 8
        BOOL = 9

    class Device:
        CPU = 0
        CUDA = 1
        METAL = 2
        VULKAN = 3

    class Layout:
        DENSE = 0
        CSR = 1
        COO = 2


DTYPE_MAP = {
    "float32": (ctypes.c_float, 4, np.float32),
    "float64": (ctypes.c_double, 8, np.float64),
    "float16": (ctypes.c_uint16, 2, np.float16),
    "bfloat16": (ctypes.c_uint16, 2, np.uint16),
    "int32": (ctypes.c_int32, 4, np.int32),
    "int64": (ctypes.c_int64, 8, np.int64),
    "int16": (ctypes.c_int16, 2, np.int16),
    "int8": (ctypes.c_int8, 1, np.int8),
    "uint8": (ctypes.c_uint8, 1, np.uint8),
    "bool": (ctypes.c_uint8, 1, np.bool_),
    "complex64": (ctypes.c_float, 8, np.complex64),
    "complex128": (ctypes.c_double, 16, np.complex128),
}

_NP_TO_DTYPE = {}
for k, (_, _, np_dt) in DTYPE_MAP.items():
    _NP_TO_DTYPE[np.dtype(np_dt)] = k


def _cuda_available() -> bool:
    """Best-effort detection of a usable CUDA device.

    Returns ``False`` when torch is unavailable or no GPU is present so that
    the (CPU-backed) simulation path is taken instead of enforcing strict
    device matching between cuda-labelled and cpu tensors.
    """
    try:
        import torch
        return bool(getattr(torch.cuda, "is_available", lambda: False)())
    except Exception:
        return False


def _resolve_dtype(dtype) -> str:
    if dtype is None:
        return "float32"
    if isinstance(dtype, str):
        return dtype
    if isinstance(dtype, np.dtype):
        return _NP_TO_DTYPE.get(dtype, "float32")
    if hasattr(dtype, "name"):
        return dtype.name.lower()
    return "float32"


def _numpy_dtype(dtype) -> np.dtype:
    return DTYPE_MAP.get(_resolve_dtype(dtype), (None, None, np.float32))[2]


class _TensorData(np.ndarray):
    """ndarray subclass that allows ``float()`` and ``item()`` on size-1 arrays."""

    def __float__(self):
        if self.size == 1:
            return float(self.item())
        return float(self.flat[0])

    def item(self, *args):
        if self.size == 1:
            return super().item(*args)
        return float(self)


class _CTensorData:
    def __init__(self, shape, dtype="float32", is_cuda=False):
        self.shape = tuple(shape)
        self.dtype = _resolve_dtype(dtype)
        self.size = math.prod(shape) if shape is not None else 0
        _, itemsize, _ = DTYPE_MAP.get(self.dtype, (None, 4, np.float32))
        self.nbytes = self.size * itemsize
        self._cptr = None
        self._is_cuda = is_cuda
        self._data = bytearray(self.nbytes)

    def _copy_from(self, arr: np.ndarray):
        if self.nbytes > 0:
            self._data[:] = arr.tobytes()

    def to_numpy(self):
        _, _, np_dt = DTYPE_MAP.get(self.dtype, (None, 4, np.float32))
        if self.nbytes == 0:
            return np.array([], dtype=np_dt).reshape(self.shape).copy()
        return np.frombuffer(self._data, dtype=np_dt).reshape(self.shape).copy()


class Tensor:
    def __init__(
        self, data=None, shape=None, dtype="float32", device="cpu", requires_grad=False
    ):
        self.device = device
        self.dtype = _resolve_dtype(dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        if isinstance(data, (int, float, np.integer, np.floating)):
            shape = shape or (1,)
            self._data = _CTensorData(
                shape, self.dtype, is_cuda=(device.startswith("cuda"))
            )
            arr = np.full(shape, float(data), dtype=_numpy_dtype(self.dtype))
            self._data._copy_from(arr)
            self.shape = shape
        elif isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=_numpy_dtype(self.dtype))
            self._data = _CTensorData(
                arr.shape, self.dtype, is_cuda=(device.startswith("cuda"))
            )
            self._data._copy_from(arr)
            self.shape = arr.shape
        elif isinstance(data, np.ndarray):
            dt = _resolve_dtype(dtype) or _resolve_dtype(str(data.dtype))
            self._data = _CTensorData(
                data.shape, dt, is_cuda=(device.startswith("cuda"))
            )
            self._data._copy_from(data.astype(_numpy_dtype(dt)))
            self.shape = tuple(data.shape)
        elif isinstance(data, Tensor):
            self._data = data._data
            self.device = data.device
            self.shape = data.shape
        else:
            shape = shape or (1,)
            self._data = _CTensorData(
                shape, self.dtype, is_cuda=(device.startswith("cuda"))
            )
            self.shape = shape

    @property
    def dtype_name(self) -> str:
        return self.dtype

    @property
    def grad_fn(self):
        return self._grad_fn

    @property
    def is_leaf(self):
        return self._grad_fn is None

    def requires_grad_(self, val=True):
        self.requires_grad = val
        return self

    def zero_grad_(self):
        self.grad = None
        return self

    def _attach_grad_fn(self, fn):
        self._grad_fn = fn

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

    @property
    def data(self) -> np.ndarray:
        return self._data.to_numpy().view(_TensorData)

    @data.setter
    def data(self, arr: np.ndarray):
        self._data = _CTensorData(
            arr.shape, _resolve_dtype(str(arr.dtype)), is_cuda=self._data._is_cuda
        )
        self._data._copy_from(arr)
        self.shape = arr.shape

    @property
    def is_cuda(self) -> bool:
        return self._data._is_cuda

    @property
    def T(self):
        if self.ndim < 2:
            return self
        return self.transpose(0, 1)

    def numpy(self) -> np.ndarray:
        return self._data.to_numpy()

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

    def _check_device_match(self, other):
        if isinstance(other, Tensor):
            dev_self = self.device.split(":")[0]
            dev_other = other.device.split(":")[0]
            # Allow mixing cuda and cpu in simulation mode (no real CUDA backend
            # or no usable GPU). Strict matching is only enforced when an actual
            # CUDA device is available.
            if dev_self != dev_other and _cuda_available():
                raise RuntimeError(
                    f"Expected all tensors to be on the same device, "
                    f"but found at least two devices: {self.device} and {other.device}"
                )

    def to(self, device: str):
        if device == self.device:
            return self
        if device.startswith("cuda") and self.device == "cpu":
            from .cuda_device import CUDAMemoryPool

            pool = CUDAMemoryPool()
            pool.allocate(self._data.nbytes)
            new_t = Tensor(self, device=self.device, dtype=self.dtype)
            new_t._data = _CTensorData(self.shape, self.dtype, is_cuda=True)
            new_t._data._data[:] = self._data._data[:]
            new_t.device = device
            return new_t
        elif device == "cpu" and self.device.startswith("cuda"):
            new_t = Tensor(self, device=self.device, dtype=self.dtype)
            new_t._data = _CTensorData(self.shape, self.dtype, is_cuda=False)
            new_t._data._data[:] = self._data._data[:]
            new_t.device = device
            return new_t
        else:
            self.device = device
            return self

    def clone(self):
        t = Tensor(self, device=self.device, dtype=self.dtype)
        t._data = _CTensorData(self.shape, self.dtype, is_cuda=self._data._is_cuda)
        t._data._data[:] = self._data._data[:]
        return t

    def item(self) -> float:
        return float(self.data.flat[0])

    def view(self, *shape):
        from .autograd_ops import Reshape

        # Handle both tuple and individual arguments
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Reshape.apply(self, shape)

    def reshape(self, *shape):
        return self.view(*shape)

    def clone(self):
        t = Tensor(self, device=self.device, dtype=self.dtype)
        t._data = _CTensorData(self.shape, self.dtype, is_cuda=self._data._is_cuda)
        t._data._data[:] = self._data._data[:]
        return t

    def copy_(self, src: "Tensor"):
        if self.shape == src.shape:
            self._data._data[:] = src._data._data[:]
        return self

    def fill_(self, value):
        arr = np.full(self.shape, value, dtype=_numpy_dtype(self.dtype))
        self._data._copy_from(arr)
        return self

    def zero_(self):
        self._data._data = bytearray(self._data.nbytes)
        return self

    def _apply_unary(self, fn: Callable):
        return Tensor(fn(self.data), dtype=self.dtype, device=self.device)

    def _apply_binary(self, other, fn: Callable):
        if isinstance(other, Tensor):
            self._check_device_match(other)
        a = self.data
        b = other.data if isinstance(other, Tensor) else other
        return Tensor(fn(a, b), dtype=self.dtype, device=self.device)

    def __add__(self, other):
        if isinstance(other, Tensor):
            self._check_device_match(other)
        from .autograd_ops import Add

        return Add.apply(self, other)

    def __radd__(self, other):
        if isinstance(other, Tensor):
            self._check_device_match(other)
        from .autograd_ops import Add

        return Add.apply(other, self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            self._check_device_match(other)
        from .autograd_ops import Sub

        return Sub.apply(self, other)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            self._check_device_match(other)
        from .autograd_ops import Sub

        return Sub.apply(other, self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            self._check_device_match(other)
        from .autograd_ops import Mul

        return Mul.apply(self, other)

    def __rmul__(self, other):
        if isinstance(other, Tensor):
            self._check_device_match(other)
        from .autograd_ops import Mul

        return Mul.apply(other, self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            self._check_device_match(other)
        from .autograd_ops import Div

        return Div.apply(self, other)

    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            self._check_device_match(other)
        from .autograd_ops import Div

        return Div.apply(other, self)

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            self._check_device_match(other)
        from .autograd_ops import MatMul

        return MatMul.apply(self, other)

    def __pow__(self, other):
        from .autograd_ops import Pow

        return Pow.apply(self, other)

    def __neg__(self):
        from .autograd_ops import Neg

        return Neg.apply(self)

    def __getitem__(self, key):
        from .autograd_ops import GetItem

        return GetItem.apply(self, key)

    def __setitem__(self, key, value):
        arr = self.data
        if isinstance(value, Tensor):
            v = value.item() if value.numel == 1 else value.data
        else:
            v = value
        arr[key] = v
        self._data._copy_from(arr)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"

    def __len__(self):
        return self.shape[0] if self.shape else 0

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def mean(self, dim=None):
        from .autograd_ops import Mean

        return Mean.apply(self, dim)

    def sum(self, dim=None):
        from .autograd_ops import Sum

        return Sum.apply(self, dim)

    def var(self, dim=None):
        if dim is None:
            return Tensor(float(self.data.var()), dtype=self.dtype)
        return Tensor(self.data.var(axis=dim), dtype=self.dtype)

    def std(self, dim=None):
        if dim is None:
            return Tensor(float(self.data.std()), dtype=self.dtype)
        return Tensor(self.data.std(axis=dim), dtype=self.dtype)

    def min(self):
        return float(self.data.min())

    def max(self):
        return float(self.data.max())

    def sqrt(self):
        from .autograd_ops import Sqrt

        return Sqrt.apply(self)

    def exp(self):
        from .autograd_ops import Exp

        return Exp.apply(self)

    def log(self):
        from .autograd_ops import Log

        return Log.apply(self)

    def abs(self):
        from .autograd_ops import Abs

        return Abs.apply(self)

    def relu(self):
        from .autograd_ops import Relu

        return Relu.apply(self)

    def sigmoid(self):
        from .autograd_ops import Sigmoid

        return Sigmoid.apply(self)

    def tanh(self):
        from .autograd_ops import Tanh

        return Tanh.apply(self)

    def tanh_act(self):
        from .autograd_ops import Tanh

        return Tanh.apply(self)

    def gelu(self):
        from .autograd_ops import Gelu

        return Gelu.apply(self)

    def silu(self):
        from .autograd_ops import Silu

        return Silu.apply(self)

    def softmax(self, dim=-1):
        from .autograd_ops import Softmax

        return Softmax.apply(self, dim)

    def log_softmax(self, dim=-1):
        from .autograd_ops import LogSoftmax

        return LogSoftmax.apply(self, dim)

    def transpose(self, dim1=0, dim2=1):
        from .autograd_ops import Transpose

        return Transpose.apply(self, dim1, dim2)

    def permute(self, *dims):
        """Permute tensor dimensions."""
        # Use numpy for permutation
        return Tensor(self.data.transpose(*dims), dtype=self.dtype, device=self.device)

    def squeeze(self, dim=None):
        from .autograd_ops import Squeeze

        return Squeeze.apply(self, dim)

    def unsqueeze(self, dim):
        from .autograd_ops import Unsqueeze

        return Unsqueeze.apply(self, dim)

    def expand(self, *shape):
        from .autograd_ops import Expand

        return Expand.apply(self, shape)

    def backward(self, grad_output=None):
        from .autograd import _wrap_tensor_backward

        _wrap_tensor_backward(self, grad_output)

    def detach(self):
        t = Tensor(self)
        t.requires_grad = False
        return t

    def save(self, path: str):
        np.save(path, self.data)

    @staticmethod
    def load(path: str):
        arr = np.load(path)
        return Tensor(arr)

    def mse_loss(self, target):
        from .autograd_ops import MSELoss

        return MSELoss.apply(self, target)

    def cross_entropy(self, target):
        from .autograd_ops import CrossEntropyLoss

        return CrossEntropyLoss.apply(self, target)

    def mae_loss(self, target):
        return Tensor(
            np.array([np.mean(np.abs(self.data - target.data))]), dtype="float32"
        )

    def nll_loss(self, target):
        from .autograd_ops import NLLLoss

        return NLLLoss.apply(self, target)

    def kl_div(self, target):
        from .autograd_ops import KLDivLoss

        return KLDivLoss.apply(self, target)

    def smooth_l1_loss(self, target, beta=1.0):
        from .autograd_ops import SmoothL1Loss

        return SmoothL1Loss.apply(self, target, beta)

    def huber_loss(self, target, delta=1.0):
        from .autograd_ops import HuberLoss

        return HuberLoss.apply(self, target, delta)

    def bce_loss(self, target):
        from .autograd_ops import BCELoss

        return BCELoss.apply(self, target)

    def bce_with_logits_loss(self, target):
        from .autograd_ops import BCEWithLogitsLoss

        return BCEWithLogitsLoss.apply(self, target)

    def binary_cross_entropy(self, target):
        from .autograd_ops import BCELoss

        return BCELoss.apply(self, target)

    def focal_loss(self, target, gamma=2.0, alpha=1.0):
        from .autograd_ops import FocalLoss

        return FocalLoss.apply(self, target, gamma, alpha)

    def triplet_margin_loss(self, positive, negative, margin=1.0):
        from .autograd_ops import TripletMarginLoss

        return TripletMarginLoss.apply(self, positive, negative, margin)

    def contrastive_loss(self, other, y, margin=1.0):
        from .autograd_ops import ContrastiveLoss

        return ContrastiveLoss.apply(self, other, y, margin)

    def cosine_embedding_loss(self, other, y, margin=0.0):
        from .autograd_ops import CosineEmbeddingLoss

        return CosineEmbeddingLoss.apply(self, other, y, margin)

    def margin_ranking_loss(self, other, y, margin=0.0):
        from .autograd_ops import MarginRankingLoss

        return MarginRankingLoss.apply(self, other, y, margin)

    def ctc_loss(
        self,
        targets,
        blank=0,
        input_lengths=None,
        target_lengths=None,
        reduction="mean",
    ):
        from .autograd_ops import CTCLoss

        return CTCLoss.apply(self, targets, blank, input_lengths, target_lengths, reduction)

    def conv1d(self, kernel, stride=1, padding=0):
        from .autograd_ops import Conv1d

        return Conv1d.apply(self, kernel, stride, padding)

    def conv2d(self, kernel, stride_h=1, stride_w=1, pad_h=0, pad_w=0):
        from scipy import signal

        arr = self.data
        k = kernel.data if isinstance(kernel, Tensor) else kernel
        if pad_h > 0 or pad_w > 0:
            arr = np.pad(
                arr,
                (
                    [(0, 0), (pad_h, pad_h), (pad_w, pad_w), (0,)]
                    if arr.ndim == 4
                    else [(pad_h, pad_h), (pad_w, pad_w)]
                ),
            )
        out = signal.correlate(arr, k, mode="valid")
        out = out[..., ::stride_h, ::stride_w]
        return Tensor(out, dtype=self.dtype)

    def pool1d(self, kernel_size, stride=None):
        stride = stride or kernel_size
        arr = self.data
        out = np.array(
            [
                arr[..., i : i + kernel_size].mean(axis=-1)
                for i in range(0, arr.shape[-1] - kernel_size + 1, stride)
            ]
        )
        if arr.ndim == 2:
            out = out.T
        return Tensor(out, dtype=self.dtype)

    def pool2d(self, kernel_h, kernel_w, stride_h=None, stride_w=None):
        stride_h = stride_h or kernel_h
        stride_w = stride_w or kernel_w
        arr = self.data
        out = np.array(
            [
                [
                    arr[..., i : i + kernel_h, j : j + kernel_w].mean(axis=(-2, -1))
                    for j in range(0, arr.shape[-1] - kernel_w + 1, stride_w)
                ]
                for i in range(0, arr.shape[-2] - kernel_h + 1, stride_h)
            ]
        )
        out = out.transpose(2, 3, 0, 1) if arr.ndim == 4 else out
        return Tensor(out, dtype=self.dtype)

    def dropout(self, rate=0.5, seed=42):
        from .autograd_ops import DropoutFn

        return DropoutFn.apply(self, rate, seed)

    def layer_norm(self, gamma, beta, eps=1e-5):
        from .autograd_ops import LayerNorm

        return LayerNorm.apply(self, gamma, beta, eps)

    def batch_norm(self, gamma, beta, running_mean, running_var, eps=1e-5):
        g = gamma.data if isinstance(gamma, Tensor) else gamma
        b = beta.data if isinstance(beta, Tensor) else beta
        rm = running_mean.data if isinstance(running_mean, Tensor) else running_mean
        rv = running_var.data if isinstance(running_var, Tensor) else running_var
        return Tensor(g * (self.data - rm) / np.sqrt(rv + eps) + b, dtype=self.dtype)

    def group_norm(self, gamma, beta, num_groups, eps=1e-5):
        g = gamma.data if isinstance(gamma, Tensor) else gamma
        b = beta.data if isinstance(beta, Tensor) else beta
        arr = self.data
        N, C, H, W = arr.shape
        arr_g = arr.reshape(N, num_groups, C // num_groups, H, W)
        mean = arr_g.mean(axis=(2, 3, 4), keepdims=True)
        var = arr_g.var(axis=(2, 3, 4), keepdims=True)
        arr_n = (arr_g - mean) / np.sqrt(var + eps)
        return Tensor(
            arr_n.reshape(N, C, H, W) * g.reshape(1, C, 1, 1) + b.reshape(1, C, 1, 1),
            dtype=self.dtype,
        )

    def embedding(self, indices):
        from .autograd_ops import EmbeddingFn

        return EmbeddingFn.apply(self, indices)

    @staticmethod
    def cat(tensors: List["Tensor"], dim=0) -> "Tensor":
        from .autograd_ops import Cat

        return Cat.apply(*tensors, dim)

    @staticmethod
    def stack(tensors: List["Tensor"], dim=0) -> "Tensor":
        from .autograd_ops import Stack

        return Stack.apply(*tensors, dim)

    @staticmethod
    def concat(tensors: List["Tensor"], dim=0) -> "Tensor":
        return Tensor.cat(tensors, dim)

    # Factory methods
    @staticmethod
    def zeros(shape, dtype="float32", device="cpu"):
        return Tensor(
            np.zeros(shape, dtype=_numpy_dtype(dtype)), dtype=dtype, device=device
        )

    @staticmethod
    def ones(shape, dtype="float32", device="cpu"):
        return Tensor(
            np.ones(shape, dtype=_numpy_dtype(dtype)), dtype=dtype, device=device
        )

    @staticmethod
    def randn(shape, dtype="float32", device="cpu"):
        return Tensor(
            np.random.randn(*shape).astype(_numpy_dtype(dtype)),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def rand(shape, dtype="float32", device="cpu"):
        return Tensor(
            np.random.rand(*shape).astype(_numpy_dtype(dtype)),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def arange(start, stop=None, step=1, dtype="float32"):
        return Tensor(
            np.arange(start, stop, step).astype(_numpy_dtype(dtype)), dtype=dtype
        )

    @staticmethod
    def eye(n, dtype="float32"):
        return Tensor(np.eye(n).astype(_numpy_dtype(dtype)), dtype=dtype)

    @staticmethod
    def full(shape, fill_value, dtype="float32"):
        return Tensor(
            np.full(shape, fill_value, dtype=_numpy_dtype(dtype)), dtype=dtype
        )

    @staticmethod
    def from_numpy(arr: np.ndarray, dtype=None):
        dt = _resolve_dtype(dtype) or _resolve_dtype(str(arr.dtype))
        return Tensor(arr, dtype=dt)


def cat(tensors: List[Tensor], dim=0) -> Tensor:
    return Tensor.cat(tensors, dim)


def stack(tensors: List[Tensor], dim=0) -> Tensor:
    return Tensor.stack(tensors, dim)


Tensorable = Union[Tensor, np.ndarray]

__all__ = [
    "Tensor",
    "Tensorable",
    "Dtype",
    "Device",
    "Layout",
    "_HAS_C_BACKEND",
    "_C_BACKEND",
    "cat",
    "stack",
]
