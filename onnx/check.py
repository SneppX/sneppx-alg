"""ONNX model checker (numpy-only).

Structural + connectivity + arity/type validation on :mod:`onnx.model` data
classes. Thin re-export of :func:`onnx.inference.check_model` plus a
convenience CLI-oriented wrapper that returns ``(ok, errors)``.
"""

from typing import List, Tuple

from .inference import check_model, infer_shapes
from .model import Model

__all__ = ["check_model", "check", "validate_model", "infer_shapes"]


def check_model(model: Model) -> Tuple[bool, List[str]]:
    """Run structural + shape/arity checks. Returns ``(ok, errors)``."""
    return __check(model)


def __check(model: Model) -> Tuple[bool, List[str]]:
    from .inference import check_model as _cm

    return _cm(model)


def check(model: Model) -> Tuple[bool, List[str]]:
    """Alias of :func:`check_model`."""
    return __check(model)


def validate_model(model: Model) -> Tuple[bool, List[str]]:
    """Alias of :func:`check_model`."""
    return __check(model)
