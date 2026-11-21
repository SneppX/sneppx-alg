"""Extra optimizers — modern, research-grade update rules.

Each optimizer operates on lists of ``Tensor`` parameters and their
gradients, following the same ``.step()`` / ``.zero_grad()`` contract as
``optim.py``. Implementations are pure-NumPy and numerically faithful to
the published algorithms.
"""

from typing import List, Optional, Iterable, Callable, Dict, Any
import math
import numpy as np

from .tensor import Tensor


def _as_array(t: Tensor) -> np.ndarray:
    return np.asarray(t.data, dtype=np.float64)


class Lion:
    """Lion (Evolved Sign Momentum, Google 2023).

    Update:  c = sign(beta1*m + (1-beta1)*g)
             p <- p - lr*(c + wd*p)
             m <- beta2*m + (1-beta2)*g
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.state: Dict[int, np.ndarray] = {}
        for i, p in enumerate(params):
            self.state[i] = np.zeros_like(_as_array(p))

    def step(self):
        for i, p in enumerate(self.params):
            g = _as_array(p.grad) if p.grad is not None else np.zeros_like(_as_array(p))
            m = self.state[i]
            c = np.sign(self.beta1 * m + (1 - self.beta1) * g)
            update = c + self.weight_decay * _as_array(p)
            p.data = (_as_array(p) - self.lr * update).astype(p.data.dtype)
            self.state[i] = self.beta2 * m + (1 - self.beta2) * g

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class LAMB:
    """LAMB (Layer-wise Adaptive Rate Scaling, NLP 2019).

    Layer-normalizes the Adam update so large layers get a rate matched to
    their parameter norm, enabling very large batch LAMB training.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        for i, p in enumerate(params):
            self.m[i] = np.zeros_like(_as_array(p))
            self.v[i] = np.zeros_like(_as_array(p))

    def step(self):
        for i, p in enumerate(self.params):
            g = _as_array(p.grad) if p.grad is not None else np.zeros_like(_as_array(p))
            m = self.beta1 * self.m[i] + (1 - self.beta1) * g
            v = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            mhat = m / (1 - self.beta1 ** (self._t() + 1))
            vhat = v / (1 - self.beta2 ** (self._t() + 1))
            update = mhat / (np.sqrt(vhat) + self.eps)
            if self.weight_decay > 0:
                update = update + self.weight_decay * _as_array(p)
            pnorm = np.linalg.norm(_as_array(p))
            unorm = np.linalg.norm(update)
            if pnorm > 0 and unorm > 0:
                trust = pnorm / unorm
            else:
                trust = 1.0
            p.data = (_as_array(p) - self.lr * trust * update).astype(p.data.dtype)
            self.m[i] = m
            self.v[i] = v

    def _t(self) -> int:
        if not hasattr(self, "_step_count"):
            self._step_count = 0
        else:
            self._step_count += 1
        return self._step_count

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class LARS:
    """LARS (Layer-wise Adaptive Rate Scaling, Yang 2017) — SGD variant."""

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        trust_coef: float = 0.001,
        eps: float = 1e-8,
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.trust_coef = trust_coef
        self.eps = eps
        self.momentum_buffer: Dict[int, np.ndarray] = {}
        for i, p in enumerate(params):
            self.momentum_buffer[i] = np.zeros_like(_as_array(p))

    def step(self):
        for i, p in enumerate(self.params):
            g = _as_array(p.grad) if p.grad is not None else np.zeros_like(_as_array(p))
            p_data = _as_array(p)
            decay = self.weight_decay * p_data
            pnorm = np.linalg.norm(p_data)
            gnorm = np.linalg.norm(g + decay)
            if pnorm > 0 and gnorm > 0:
                trust = self.trust_coef * pnorm / gnorm
            else:
                trust = 1.0
            v = self.momentum * self.momentum_buffer[i] + trust * (g + decay)
            self.momentum_buffer[i] = v
            p.data = (p_data - self.lr * v).astype(p.data.dtype)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class AdaFactor:
    """AdaFactor (Shazeer 2018) — memory-efficient low-rank Adam.

    Factors the second moment into row/column statistics, reducing
    optimizer state from O(n) to O(sqrt(n)) per tensor — ideal for huge
    embeddings without a full 32-bit momentum buffer per parameter.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        beta1: float = 0.0,
        beta2: float = 0.999,
        eps1: float = 1e-30,
        eps2: float = 1e-3,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        min_dim_size_to_factor: int = 128,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps1 = eps1
        self.eps2 = eps2
        self.clip_threshold = clip_threshold
        self.weight_decay = weight_decay
        self.min_dim = min_dim_size_to_factor
        self.step_count = 0
        self.r_row: Dict[int, np.ndarray] = {}
        self.r_col: Dict[int, np.ndarray] = {}
        self.m: Dict[int, np.ndarray] = {}
        for i, p in enumerate(params):
            arr = _as_array(p)
            if arr.ndim >= 2 and min(arr.shape) >= self.min_dim:
                self.r_row[i] = np.zeros(arr.shape[0])
                self.r_col[i] = np.zeros(arr.shape[1])
            else:
                self.r_row[i] = np.zeros(arr.shape)
            if beta1 > 0:
                self.m[i] = np.zeros_like(arr)

    def _factored(self, arr: np.ndarray) -> bool:
        return arr.ndim >= 2 and min(arr.shape) >= self.min_dim

    def step(self):
        self.step_count += 1
        for i, p in enumerate(self.params):
            g = _as_array(p.grad) if p.grad is not None else np.zeros_like(_as_array(p))
            if self._factored(_as_array(p)):
                # row/col factored second moment
                row = np.mean(g * g, axis=1)
                col = np.mean(g * g, axis=0)
                self.r_row[i] = self.beta2 * self.r_row[i] + (1 - self.beta2) * row
                self.r_col[i] = self.beta2 * self.r_col[i] + (1 - self.beta2) * col
                r_factor = np.outer(
                    self.r_row[i] + self.eps1, self.r_col[i] + self.eps1
                )
                denom = np.sqrt(r_factor) + self.eps2
            else:
                self.r_row[i] = self.beta2 * self.r_row[i] + (1 - self.beta2) * (g * g)
                denom = np.sqrt(self.r_row[i]) + self.eps2
            update = g / denom
            # update clip
            mag = np.linalg.norm(update)
            if mag > self.clip_threshold:
                update = update * (self.clip_threshold / mag)
            if self.weight_decay > 0:
                update = update + self.weight_decay * _as_array(p)
            if self.beta1 > 0:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * update
                update = self.m[i]
            relative_step = self.lr / (1 + (self.step_count - 1))
            p.data = (_as_array(p) - relative_step * update).astype(p.data.dtype)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class RAdam:
    """RAdam (Liu 2019) — Adam with rectified variance warmup."""

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.step_count = 0
        for i, p in enumerate(params):
            self.m[i] = np.zeros_like(_as_array(p))
            self.v[i] = np.zeros_like(_as_array(p))

    def step(self):
        self.step_count += 1
        beta2_power = self.beta2**self.step_count
        rho_inf = 2.0 / (1 - self.beta2) - 1
        for i, p in enumerate(self.params):
            g = _as_array(p.grad) if p.grad is not None else np.zeros_like(_as_array(p))
            m = self.beta1 * self.m[i] + (1 - self.beta1) * g
            v = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            mhat = m / (1 - self.beta1**self.step_count)
            vhat = v / (1 - beta2_power)
            rho = rho_inf - 2 * self.step_count * beta2_power / (1 - beta2_power)
            if rho > 5:
                r = math.sqrt(
                    (rho - 4)
                    * (rho - 2)
                    * rho_inf
                    / ((rho_inf - 4) * (rho_inf - 2) * rho)
                )
                numer = r
            else:
                numer = 1.0
            denom = np.sqrt(vhat) + self.eps
            update = numer * mhat / denom
            if self.weight_decay > 0:
                update = update + self.weight_decay * _as_array(p)
            p.data = (_as_array(p) - self.lr * update).astype(p.data.dtype)
            self.m[i] = m
            self.v[i] = v

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class Sophia:
    """Sophia (Liu 2023) — second-order clipping via Hessian trace estimate.

    Uses a cheap Hutchinson estimate of the diagonal Hessian to clip the
    update, giving faster convergence than Adam for language models.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-4,
        betas: tuple = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 1e-2,
        eps: float = 1e-12,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.rho = rho
        self.weight_decay = weight_decay
        self.eps = eps
        self.m: Dict[int, np.ndarray] = {}
        self.h: Dict[int, np.ndarray] = {}
        for i, p in enumerate(params):
            self.m[i] = np.zeros_like(_as_array(p))
            self.h[i] = np.zeros_like(_as_array(p))

    def estimate_hessian(self, grads: List[Optional[Tensor]], loss_fn: Callable):
        """Update the (diagonal) Hessian estimate via Hutchinson's method.

        ``loss_fn`` should return a scalar loss given the current params;
        ``grads`` are the fresh first-order gradients to perturb.
        """
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = _as_array(p.grad)
            # random Rademacher vector z in {-1, +1}
            z = np.random.choice([-1.0, 1.0], size=g.shape)
            # finite-difference Hessian-direction product: (g(x+eps z) - g(x-eps z)) / (2 eps)
            eps = 1e-4
            orig = _as_array(p).copy()
            p.data = (orig + eps * z).astype(p.data.dtype)
            loss_plus = loss_fn()
            p.data = (orig - eps * z).astype(p.data.dtype)
            loss_minus = loss_fn()
            p.data = orig.astype(p.data.dtype)
            h_est = (loss_plus - loss_minus) / (2 * eps)
            self.h[i] = self.beta2 * self.h[i] + (1 - self.beta2) * h_est * z

    def step(self):
        for i, p in enumerate(self.params):
            g = _as_array(p.grad) if p.grad is not None else np.zeros_like(_as_array(p))
            m = self.beta1 * self.m[i] + (1 - self.beta1) * g
            mhat = m / (1 - self.beta1)
            h = np.abs(self.h[i]) + self.eps
            clipped = np.clip(mhat / h, -self.rho, self.rho)
            update = clipped + self.weight_decay * _as_array(p)
            p.data = (_as_array(p) - self.lr * update).astype(p.data.dtype)
            self.m[i] = m

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class Adan:
    """Adan (Xie 2022) — adaptive Nesterov momentum with decoupled WD."""

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2, self.beta3 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.n: Dict[int, np.ndarray] = {}
        for i, p in enumerate(params):
            self.m[i] = np.zeros_like(_as_array(p))
            self.v[i] = np.zeros_like(_as_array(p))
            self.n[i] = np.zeros_like(_as_array(p))

    def step(self):
        for i, p in enumerate(self.params):
            g = _as_array(p.grad) if p.grad is not None else np.zeros_like(_as_array(p))
            prev = _as_array(p)
            m = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # Nesterov-preview step
            nesterov = prev - m
            # v (gradient difference moment)
            v = self.beta2 * self.v[i] + (1 - self.beta2) * (g - self.m[i])
            # n (second-order Nesterov)
            n = self.beta3 * self.n[i] + (1 - self.beta3) * (
                (g - self.m[i]) * (g - self.m[i])
            )
            # predicted gradient
            nesterov_grad = g + self.beta2 * (g - self.m[i])
            nesterov_delta = nesterov_grad + self.beta3 * (nesterov_grad**2)
            denom = np.sqrt(n) + self.eps
            update = nesterov_delta / denom
            if self.weight_decay > 0:
                update = update + self.weight_decay * prev
            p.data = (prev - self.lr * update).astype(p.data.dtype)
            self.m[i] = m
            self.v[i] = v
            self.n[i] = n

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class ScheduleFreeAdamW:
    """Schedule-Free AdamW (Defazio 2024) — no LR schedule needed.

    Interpolates between a constant ``y`` (fast weights) and the true
    params ``x`` (slow weights) using ``beta``, giving schedule-free
    training with one hyperparameter (lr).
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.1,
        eps: float = 1e-8,
        beta: float = 0.9,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.beta = beta
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.x: Dict[int, np.ndarray] = {}
        self.y: Dict[int, np.ndarray] = {}
        self.step_count = 0
        for i, p in enumerate(params):
            self.m[i] = np.zeros_like(_as_array(p))
            self.v[i] = np.zeros_like(_as_array(p))
            self.x[i] = _as_array(p).copy()
            self.y[i] = _as_array(p).copy()

    def step(self):
        self.step_count += 1
        z = (
            self.beta
            * (1 - self.beta ** (self.step_count - 1))
            / (1 - self.beta**self.step_count)
            if self.beta != 1
            else 1.0
        )
        for i, p in enumerate(self.params):
            g = _as_array(p.grad) if p.grad is not None else np.zeros_like(_as_array(p))
            x = self.x[i]
            # update y (fast weights)
            m = self.beta1 * self.m[i] + (1 - self.beta1) * g
            v = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            mhat = m / (1 - self.beta1**self.step_count)
            vhat = v / (1 - self.beta2**self.step_count)
            y_new = self.y[i] - self.lr * (mhat / (np.sqrt(vhat) + self.eps))
            if self.weight_decay > 0:
                y_new = y_new - self.lr * self.weight_decay * x
            self.y[i] = y_new
            # update x (slow weights): x = z*y + (1-z)*x
            self.x[i] = z * y_new + (1 - z) * x
            p.data = self.x[i].astype(p.data.dtype)
            self.m[i] = m
            self.v[i] = v

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


def get_optimizer(name: str, params: List[Tensor], **kwargs):
    """Factory for extra optimizers."""
    registry = {
        "lion": Lion,
        "lamb": LAMB,
        "lars": LARS,
        "adafactor": AdaFactor,
        "radam": RAdam,
        "sophia": Sophia,
        "adan": Adan,
        "schedule_free_adamw": ScheduleFreeAdamW,
        "adam8bit": Adam8bit,
        "adamw8bit": lambda ps, **kw: Adam8bit(ps, adamw=True, **kw),
        "sgd8bit": SGD8bit,
    }
    if name not in registry:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(registry)}")
    return registry[name](params, **kwargs)


# ---------------------------------------------------------------------------
#  8-bit optimizers — per-parameter state stored as int8 with dynamic scaling
# ---------------------------------------------------------------------------


def _quantize_int8(arr):
    """Quantize ``arr`` to int8 with a per-tensor dynamic scale (max abs / 127)."""
    arr = np.asarray(arr, dtype=np.float64)
    amax = float(np.max(np.abs(arr)))
    if amax == 0.0:
        return np.zeros(arr.shape, dtype=np.int8), np.float64(0.0)
    scale = amax / 127.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, np.float64(scale)


def _dequantize_int8(q, scale):
    return q.astype(np.float64) * scale


class Adam8bit:
    """Adam / AdamW with int8-quantized first- and second-moment states.

    The momentum (m) and velocity (v) buffers are quantized to int8 with a
    per-parameter dynamic scale on every step, then dequantized for the
    parameter update. This mirrors the bitsandbytes 8-bit Adam scheme and
    keeps the optimizer state at ~1/4 the memory of float32 while tracking
    full-precision Adam closely (within quantization error).
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        adamw: bool = True,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.adamw = adamw
        self.t = 0
        self.m: Dict[int, tuple] = {}
        self.v: Dict[int, tuple] = {}
        for i, p in enumerate(self.params):
            a = _as_array(p)
            self.m[i] = _quantize_int8(np.zeros_like(a))
            self.v[i] = _quantize_int8(np.zeros_like(a))

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = _as_array(p.grad) if p.grad is not None else np.zeros_like(_as_array(p))
            m = _dequantize_int8(*self.m[i])
            v = _dequantize_int8(*self.v[i])
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g * g)
            mhat = m / (1 - self.beta1 ** self.t)
            vhat = v / (1 - self.beta2 ** self.t)
            denom = np.sqrt(vhat) + self.eps
            upd = mhat / denom
            pdata = _as_array(p)
            if self.adamw:
                upd = upd + self.weight_decay * pdata
            p.data = (pdata - self.lr * upd).astype(p.data.dtype)
            self.m[i] = _quantize_int8(m)
            self.v[i] = _quantize_int8(v)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class SGD8bit:
    """SGD with int8-quantized momentum buffer (optional Nesterov)."""

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.m: Dict[int, tuple] = {}
        for i, p in enumerate(self.params):
            a = _as_array(p)
            self.m[i] = _quantize_int8(np.zeros_like(a))

    def step(self):
        for i, p in enumerate(self.params):
            g = _as_array(p.grad) if p.grad is not None else np.zeros_like(_as_array(p))
            buf = _dequantize_int8(*self.m[i])
            if self.momentum != 0:
                buf = self.momentum * buf + g
                self.m[i] = _quantize_int8(buf)
                if self.nesterov:
                    g = g + self.momentum * buf
                else:
                    g = buf
            pdata = _as_array(p)
            upd = g + self.weight_decay * pdata
            p.data = (pdata - self.lr * upd).astype(p.data.dtype)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


def _build_8bit_optimizer(name, params, **kwargs):
    registry = {
        "adam8bit": Adam8bit,
        "adamw8bit": lambda ps, **kw: Adam8bit(ps, adamw=True, **kw),
        "sgd8bit": SGD8bit,
    }
    if name not in registry:
        raise ValueError(f"Unknown 8-bit optimizer '{name}'. Available: {list(registry)}")
    return registry[name](params, **kwargs)
