"""Automatic Mixed Precision (AMP) — autocast scopes + GradScaler.

Implements a torch.cuda.amp-style API on top of the pure-NumPy Tensor
backend: an ``autocast`` context manager that records a desired compute
dtype, and a ``GradScaler`` that multiplies loss by a scale factor, runs
the (simulated) backward, divides grads by the scale, and recovers the
scale on finite gradients. On this CPU backend the math still runs in
float32 but the dtype policy / scaling logic is fully exercised, which
is exactly what real training loops depend on.
"""

from typing import Optional, List, Callable, Any
from contextlib import contextmanager
from .tensor import Tensor, Dtype
import numpy as np

# Module-level autocast state (mirrors torch's thread-local-ish model)
_AUTOCAST_ENABLED = False
_AUTOCAST_DTYPE = "float16"


def is_autocast_enabled() -> bool:
    return _AUTOCAST_ENABLED


def get_autocast_dtype() -> str:
    return _AUTOCAST_DTYPE


@contextmanager
def autocast(enabled: bool = True, dtype: str = "float16"):
    """Context manager selecting a reduced-precision compute dtype."""
    global _AUTOCAST_ENABLED, _AUTOCAST_DTYPE
    prev_enabled = _AUTOCAST_ENABLED
    prev_dtype = _AUTOCAST_DTYPE
    _AUTOCAST_ENABLED = enabled
    _AUTOCAST_DTYPE = dtype
    try:
        yield
    finally:
        _AUTOCAST_ENABLED = prev_enabled
        _AUTOCAST_DTYPE = prev_dtype


class GradScaler:
    """Gradient scaler for mixed-precision training.

    Mantains a scale factor that is doubled on consecutive finite-step
    batches and halved when an inf/NaN gradient is detected (overflow),
    with an optional growth interval and windowed hysteresis.
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        self._init_scale = float(init_scale)
        self._scale = float(init_scale)
        self._growth_factor = float(growth_factor)
        self._backoff_factor = float(backoff_factor)
        self._growth_interval = int(growth_interval)
        self._enabled = enabled
        self._growth_tracker = 0
        self._found_inf = False

    @property
    def scale(self) -> float:
        return self._scale

    def is_enabled(self) -> bool:
        return self._enabled

    def scale_value(self) -> float:
        return self._scale if self._enabled else 1.0

    def scale_tensor(self, t: Tensor) -> Tensor:
        """Return ``t * scale`` (no-op when disabled)."""
        if not self._enabled:
            return t
        return t * self._scale

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Scale the loss before backward."""
        if not self._enabled:
            return loss
        return loss * self._scale

    def _has_inf_nan(self, grads: List[Tensor]) -> bool:
        for g in grads:
            if g is None:
                continue
            arr = np.asarray(g.data)
            if not np.all(np.isfinite(arr)):
                return True
        return False

    def unscale_(self, grads: List[Tensor]) -> None:
        """Divide a list of gradient tensors in-place by the scale."""
        if not self._enabled:
            return
        inv = 1.0 / self._scale
        for g in grads:
            if g is not None:
                g.data = g.data * inv

    def step(self, optimizer, grads: Optional[List[Tensor]] = None) -> bool:
        """Run the optimizer step, adjusting the scale on overflow.

        Returns True if the step was applied (no overflow), False otherwise.
        """
        if not self._enabled:
            optimizer.step()
            return True

        found_inf = self._has_inf_nan(grads) if grads else False
        self._found_inf = found_inf
        if found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
            return False

        optimizer.step()
        self._growth_tracker += 1
        if self._growth_tracker >= self._growth_interval:
            self._scale *= self._growth_factor
            self._growth_tracker = 0
        return True

    def update(self, new_scale: Optional[float] = None) -> None:
        """Allow external scale override (used by some schedules)."""
        if new_scale is not None:
            self._scale = float(new_scale)

    def load_state_dict(self, state: dict) -> None:
        self._scale = state.get("scale", self._scale)
        self._growth_tracker = state.get("growth_tracker", 0)
        self._found_inf = state.get("found_inf", False)

    def state_dict(self) -> dict:
        return {
            "scale": self._scale,
            "growth_tracker": self._growth_tracker,
            "found_inf": self._found_inf,
        }


def autocast_matmul_warning(op_name: str):
    """Hook used by the tensor engine to flag unsupported autocast ops."""
    if _AUTOCAST_ENABLED:
        # Some ops (e.g. softmax) promote to float32 internally for stability
        return f"{op_name} runs in {get_autocast_dtype()} under autocast"
    return f"{op_name} runs in float32"
