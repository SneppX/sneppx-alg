"""Weight-decay helpers — coupled (in-gradient) and decoupled (AdamW-style)."""

import numpy as np


def _a(x) -> np.ndarray:
    if hasattr(x, "data"):
        x = x.data
    return np.asarray(x, dtype=np.float64)


def add_l2_grad(param, grad, weight_decay: float) -> np.ndarray:
    """Coupled L2: add ``weight_decay * param`` to the gradient."""
    return _a(grad) + weight_decay * _a(param)


def apply_weight_decay(param, grad, lr: float, weight_decay: float) -> np.ndarray:
    """One coupled-SGD-with-momentum-free step: ``p - lr * (g + wd*p)``."""
    return _a(param) - lr * (_a(grad) + weight_decay * _a(param))


def apply_decoupled_weight_decay(
    param, grad, lr: float, weight_decay: float, step_fn=None
) -> np.ndarray:
    """Decoupled weight decay (AdamW): shrink then take the optimizer step.

    ``step_fn(param, grad, lr)`` performs the gradient part; when omitted a
    plain SGD step ``p - lr*g`` is used.
    """
    p = _a(param)
    if step_fn is None:
        stepped = p - lr * _a(grad)
    else:
        stepped = _a(step_fn(p, _a(grad), lr))
    return stepped - lr * weight_decay * p
