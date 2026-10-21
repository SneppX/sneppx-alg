# -*- coding: utf-8 -*-
"""c_tensor_bridge — numpy <-> C `_Tensor` (true C tensor engine) bridge.

# WHAT
   Two-way conversion helpers between ``numpy.ndarray`` and the genuine C
   tensor engine exposed by the compiled extension (``SneppX_ALG._SNEPPX_c``
   — pybind class ``_Tensor`` over ``SNEPPXTensor``), plus thin conveniences
   and a parity-check helper.

# WHY
   Issue #14: the C ``_Tensor`` engine is implemented in C11 with its own
   ops (``sin``, ``matmul``, ``conv2d``, ``pool2d``, norm ops, ...) but was
   unreachable from the public Python ``Tensor`` class (which is a NumPy
   simulation).  This bridge makes the C engine usable directly, letting
   Python tests exercise the real C compute paths and letting C results be
   inspected with NumPy.

# HOW
   The pybind API is deliberately primitive:
   - ``_Tensor.create(shape1d, dtype)`` allocates an *uninitialized* tensor
     of the given 1-D shape vector (it does NOT import data),
   - ``t.set_data(arr)`` memcpy's a flat buffer (byte-sizemust match),
   - ``t.data()`` returns a numpy view backed by C memory.
   So ``from_numpy`` = create + set_data, ``to_numpy`` = np.array(data()).
"""

import numpy as np

from .. import _neural_engine_bridge as _ext
from .. import _neural_engine_bridge as _C_BACKEND  # canonical alias

_Tensor = None


def available() -> bool:
    """True when the compiled C tensor engine can be imported."""
    global _Tensor
    if _Tensor is not None:
        return True
    try:
        _Tensor = _ext._Tensor
        _ = _ext.SNEPPXDtype
        return True
    except Exception:
        return False


def _get_Tensor():
    if _Tensor is None and not available():
        raise RuntimeError("C tensor engine not built — compile _SNEPPX_c first")
    return _Tensor


# ---------------------------------------------------------------------------
# dtype mapping
# ---------------------------------------------------------------------------


def numpy_dtype_to_c(np_dtype) -> int:
    """Map a numpy dtype (or python scalar) to ``SNEPPXDtype`` enum."""
    dt = np.dtype(np_dtype)
    table = {
        np.float32: "f32",
        np.float64: "f64",
        np.float16: "f16",
        np.int32: "i32",
        np.int64: "i64",
        np.int16: "i16",
        np.int8: "i8",
        np.uint8: "u8",
        np.bool_: "bool",
        np.complex64: "c64",
        np.complex128: "c128",
    }
    name = table.get(dt.type)
    D = _ext.SNEPPXDtype
    if name in ("f32",):
        return D.FLOAT32
    if name in ("f64",):
        return D.FLOAT64
    if name in ("f16",):
        return D.FLOAT16
    if name in ("i32",):
        return D.INT32
    if name in ("i64",):
        return D.INT64
    if name in ("i16",):
        return D.INT16
    if name in ("i8",):
        return D.INT8
    if name in ("u8",):
        return D.UINT8
    if name in ("bool",):
        return D.BOOL
    if name in ("c64",):
        return D.COMPLEX64
    if name in ("c128",):
        return D.COMPLEX128
    raise TypeError(f"unsupported numpy dtype for C tensor: {np_dtype!r}")


def c_dtype_to_numpy(c_dtype) -> np.dtype:
    """Map a ``SNEPPXDtype`` enum to the matching numpy dtype."""
    D = _ext.SNEPPXDtype
    table = {
        D.FLOAT32: np.float32,
        D.FLOAT64: np.float64,
        D.FLOAT16: np.float16,
        D.INT32: np.int32,
        D.INT64: np.int64,
        D.INT16: np.int16,
        D.INT8: np.int8,
        D.UINT8: np.uint8,
        D.BOOL: np.bool_,
        D.COMPLEX64: np.complex64,
        D.COMPLEX128: np.complex128,
    }
    if c_dtype not in table:
        raise TypeError(f"unsupported C dtype: {c_dtype!r}")
    return np.dtype(table[c_dtype])


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------


def from_numpy(arr, dtype=None):
    """Create a C ``_Tensor`` with the contents (and shape) of ``arr``.

    Copies: the C tensor owns its memory once this returns.
    ``dtype`` may be a ``SNEPPXDtype`` or a numpy/python dtype; default is
    ``arr``'s own dtype (float32/float64/etc. preserved when supported,
    else float32).
    """
    T = _get_Tensor()
    a = np.ascontiguousarray(arr)
    if dtype is None:
        try:
            dtype = numpy_dtype_to_c(a.dtype)
        except TypeError:
            dtype = _ext.SNEPPXDtype.FLOAT32
            a = a.astype(np.float32)
    elif not hasattr(dtype, "value"):  # plain numpy dtype given
        dtype = numpy_dtype_to_c(dtype)
    shape = np.asarray(a.shape, dtype=np.int64)
    t = T.create(shape, dtype)
    flat = np.ravel(a).view(a.dtype) if a.dtype.itemsize else np.ravel(a)
    # set_data requires the numpy buffer byte size to equal the tensor's
    t.set_data(flat)
    return t


def to_numpy(ct) -> np.ndarray:
    """Copy a C ``_Tensor`` into a fresh numpy array (C memory independent)."""
    view = ct.data()  # numpy view backed by C memory
    return np.array(view, copy=True)


def round_trip(arr, dtype=None) -> np.ndarray:
    """numpy -> C -> numpy; returns what the C engine stored."""
    return to_numpy(from_numpy(arr, dtype))


# ---------------------------------------------------------------------------
# quick parity helpers (used by the tests and handy for debugging)
# ---------------------------------------------------------------------------


def parity_numpy_vs_c_add(a, b, atol=1e-6) -> float:
    """Max abs error between numpy add and C add for same inputs."""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    ta = from_numpy(a)
    tb = from_numpy(b)
    got = to_numpy(ta.add(tb))
    return float(np.max(np.abs(got - (a + b))))


def parity_numpy_vs_c_matmul(a, b, atol=1e-4) -> float:
    """Max abs error between numpy matmul and C matmul for same inputs."""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    ta = from_numpy(a)
    tb = from_numpy(b)
    got = to_numpy(ta.matmul(tb))
    return float(np.max(np.abs(got - (a @ b))))
