"""SneppX-ALG — Python bindings for the cognitive processing system.

Usage:
    from SneppX_ALG import Tensor, Linear, AdamW, Trainer
    from SneppX_ALG import _neural_engine_bridge as ax

    t = Tensor.zeros([4, 8], ax.SNEPPXDtype.FLOAT32)
"""

try:
    from . import _arix_c as _neural_engine_bridge
except ImportError:
    try:
        from . import _SNEPPX_c as _neural_engine_bridge
    except ImportError:
        import types

        _neural_engine_bridge = types.ModuleType("_neural_engine_bridge_fallback")

from .interface_bindings import *
from . import metrics
from . import datasets
from . import sampling
from . import regularization
from . import callbacks
from . import linalg
from . import signal
from . import graph
from . import stats
from . import structured_output

__all__ = [
    "_neural_engine_bridge",
    "metrics",
    "datasets",
    "sampling",
    "regularization",
    "callbacks",
    "linalg",
    "signal",
    "graph",
    "stats",
    "structured_output",
]


def __getattr__(name):
    if name == "onnx":
        # standalone numpy-only ONNX toolkit in the top-level onnx/ package
        import os as _os
        import sys as _sys

        _root = _os.path.dirname(
            _os.path.dirname(
                _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            )
        )
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        import importlib as _importlib

        _mod = _importlib.import_module("onnx")
        globals()["onnx"] = _mod
        return _mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
