"""onnxruntime adapter (optional dependency).

Provides an ``OnnxRuntimeSession`` wrapper that uses the official
``onnxruntime`` package when installed, and falls back to the pure-numpy
executor otherwise. Install with ``pip install onnxruntime`` for full op
coverage and hardware acceleration.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ..model import Model
from ..parser import parse_model
from ..serializer import serialize_model

__all__ = ["OnnxRuntimeSession", "has_onnxruntime", "check_onnxruntime"]


def has_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


def check_onnxruntime() -> str:
    """Return the onnxruntime version or a clear error message."""
    if not has_onnxruntime():
        return "onnxruntime not installed (pip install onnxruntime)"
    import onnxruntime

    return onnxruntime.__version__


class OnnxRuntimeSession:
    """Session backed by onnxruntime when available.

    If ``onnxruntime`` is missing, falls back to the numpy executor so callers
    can always construct a session; set ``require_ort=True`` to raise instead.
    """

    def __init__(self, model: Model, require_ort: bool = False):
        self.model = model
        if has_onnxruntime():
            import onnxruntime as ort

            self._ort = ort
            self._session = ort.InferenceSession(serialize_model(model))
        else:
            if require_ort:
                raise ImportError(
                    "onnxruntime is required: pip install onnxruntime"
                )
            from .numpy_executor import Session as NumpySession

            self._session = NumpySession(model)

    def get_inputs(self):
        return self._session.get_inputs()

    def get_outputs(self):
        return self._session.get_outputs()

    def run(self, inputs: Dict[str, np.ndarray], output_names: Optional[List[str]] = None):
        if self._ort is not None:
            names = output_names if output_names is not None else [
                o.name for o in self._session.get_outputs()
            ]
            return self._session.run(names, inputs)
        return self._session.run(inputs)
