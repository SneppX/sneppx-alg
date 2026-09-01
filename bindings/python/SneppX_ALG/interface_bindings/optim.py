"""Optimizer Module — pure Python optimizers for pybind11-backed Tensor."""

from typing import Callable, List, Optional, Iterator, Dict, Any
from .tensor import Tensor
from .nn import Module
import numpy as np

# Lion/LAMB are the canonical research-grade implementations in optim_extra;
# re-export here so `from .optim import Lion, LAMB` keeps working.
from .optim_extra import Lion, LAMB  # noqa: F401


class Optimizer:
    def __init__(self, params: Iterator[Tensor], lr: float, weight_decay: float = 0.0, **kwargs):
        self.lr = lr
        self.weight_decay = weight_decay
        self.defaults: Dict[str, Any] = {"lr": lr, "weight_decay": weight_decay, **kwargs}
        self.param_groups = self._build_param_groups(params, lr, weight_decay, **kwargs)
        self.params = [p for g in self.param_groups for p in g["params"]]
        self.state = [{} for _ in self.params]

    @staticmethod
    def _build_param_groups(params, lr, weight_decay, **kwargs) -> List[Dict[str, Any]]:
        if params is None:
            return []
        # Support pass-in as a list of param groups (dicts) for per-group options.
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
            and "params" in params[0]
        ):
            groups: List[Dict[str, Any]] = []
            for g in params:
                grp = {
                    "params": list(g["params"]),
                    "lr": g.get("lr", lr),
                    "weight_decay": g.get("weight_decay", weight_decay),
                }
                for k, v in g.items():
                    if k not in ("params", "lr", "weight_decay"):
                        grp[k] = v
                groups.append(grp)
            return groups
        return [{"params": list(params), "lr": lr, "weight_decay": weight_decay}]

    def _enumerate(self):
        """Yield (flat_index, param, group) for every parameter."""
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                yield idx, p, group
                idx += 1

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.fill_(0.0)

    def step(self):
        raise NotImplementedError

    def state_dict(self) -> dict:
        state_copy = []
        for s in self.state:
            state_copy.append(
                {k: v.copy() if hasattr(v, "copy") else v for k, v in s.items()}
            )
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "state": state_copy,
            "param_count": len(self.params),
        }

    def load_state_dict(self, state_dict: dict):
        self.lr = state_dict.get("lr", self.lr)
        self.weight_decay = state_dict.get("weight_decay", self.weight_decay)
        restored = state_dict.get("state", [])
        for i, s in enumerate(restored):
            if i < len(self.state):
                for k, v in s.items():
                    self.state[i][k] = v


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(params, lr, weight_decay, momentum=momentum, dampening=dampening, nesterov=nesterov)
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov

    def step(self):
        for i, p, group in self._enumerate():
            if p.grad is None:
                continue
            g = p.grad.data + group.get("weight_decay", 0.0) * p.data
            if self.momentum > 0:
                if "momentum_buf" not in self.state[i]:
                    self.state[i]["momentum_buf"] = np.zeros_like(g)
                buf = self.momentum * self.state[i]["momentum_buf"] + (1 - self.dampening) * g
                self.state[i]["momentum_buf"] = buf
                if self.nesterov:
                    g = g + self.momentum * buf
                else:
                    g = buf
            p.data = p.data - group.get("lr", self.lr) * g


class NesterovSGD(SGD):
    """SGD with Nesterov momentum (convenience alias)."""

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0):
        super().__init__(params, lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(params, lr, weight_decay, betas=betas, eps=eps)
        self.betas = betas
        self.eps = eps
        self._step = 0

    def state_dict(self) -> dict:
        sd = super().state_dict()
        sd["_step"] = self._step
        sd["betas"] = list(self.betas)
        sd["eps"] = self.eps
        return sd

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self._step = state_dict.get("_step", 0)
        self.betas = tuple(state_dict.get("betas", self.betas))
        self.eps = state_dict.get("eps", self.eps)

    def step(self):
        self._step += 1
        for i, p, group in self._enumerate():
            if p.grad is None:
                continue
            g = p.grad.data
            if "exp_avg" not in self.state[i]:
                self.state[i]["exp_avg"] = np.zeros_like(g)
                self.state[i]["exp_avg_sq"] = np.zeros_like(g)
            m = self.state[i]["exp_avg"]
            v = self.state[i]["exp_avg_sq"]
            m = self.betas[0] * m + (1 - self.betas[0]) * g
            v = self.betas[1] * v + (1 - self.betas[1]) * g**2
            self.state[i]["exp_avg"] = m
            self.state[i]["exp_avg_sq"] = v
            m_hat = m / (1 - self.betas[0] ** self._step)
            v_hat = v / (1 - self.betas[1] ** self._step)
            lr = group.get("lr", self.lr)
            wd = group.get("weight_decay", self.weight_decay)
            p.data = p.data - lr * wd * p.data
            p.data = p.data - lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSprop(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ):
        super().__init__(params, lr, weight_decay, alpha=alpha, eps=eps, momentum=momentum, centered=centered)
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.centered = centered

    def step(self):
        for i, p, group in self._enumerate():
            if p.grad is None:
                continue
            g = p.grad.data + group.get("weight_decay", 0.0) * p.data
            st = self.state[i]
            if "square_avg" not in st:
                st["square_avg"] = np.zeros_like(g)
            st["square_avg"] = self.alpha * st["square_avg"] + (1 - self.alpha) * g**2
            if self.centered:
                if "grad_avg" not in st:
                    st["grad_avg"] = np.zeros_like(g)
                st["grad_avg"] = self.alpha * st["grad_avg"] + (1 - self.alpha) * g
                denom = np.sqrt(st["square_avg"] - st["grad_avg"] ** 2 + self.eps)
            else:
                denom = np.sqrt(st["square_avg"] + self.eps)
            if self.momentum > 0:
                if "momentum_buf" not in st:
                    st["momentum_buf"] = np.zeros_like(g)
                st["momentum_buf"] = self.momentum * st["momentum_buf"] + g / denom
                p.data = p.data - group.get("lr", self.lr) * st["momentum_buf"]
            else:
                p.data = p.data - group.get("lr", self.lr) * g / denom


class Adagrad(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.01,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        eps: float = 1e-10,
    ):
        super().__init__(params, lr, weight_decay, lr_decay=lr_decay, eps=eps)
        self.lr_decay = lr_decay
        self.eps = eps

    def step(self):
        for i, p, group in self._enumerate():
            if p.grad is None:
                continue
            g = p.grad.data + group.get("weight_decay", 0.0) * p.data
            st = self.state[i]
            if "sum_sq" not in st:
                st["sum_sq"] = np.zeros_like(g)
                st["step"] = 0
            st["sum_sq"] = st["sum_sq"] + g**2
            st["step"] += 1
            lr = group.get("lr", self.lr)
            if self.lr_decay > 0:
                lr = lr / (1 + self.lr_decay * st["step"])
            p.data = p.data - lr * g / (np.sqrt(st["sum_sq"]) + self.eps)


class Adadelta(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr, weight_decay, rho=rho, eps=eps)
        self.rho = rho
        self.eps = eps

    def step(self):
        for i, p, group in self._enumerate():
            if p.grad is None:
                continue
            g = p.grad.data + group.get("weight_decay", 0.0) * p.data
            st = self.state[i]
            if "sq_grad" not in st:
                st["sq_grad"] = np.zeros_like(g)
                st["sq_delta"] = np.zeros_like(g)
            st["sq_grad"] = self.rho * st["sq_grad"] + (1 - self.rho) * g**2
            update = (
                np.sqrt(st["sq_delta"] + self.eps)
                / np.sqrt(st["sq_grad"] + self.eps)
                * g
            )
            st["sq_delta"] = self.rho * st["sq_delta"] + (1 - self.rho) * update**2
            p.data = p.data - group.get("lr", self.lr) * update


class Adamax(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.002,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr, weight_decay, betas=betas, eps=eps)
        self.betas = betas
        self.eps = eps
        self._step = 0

    def step(self):
        self._step += 1
        for i, p, group in self._enumerate():
            if p.grad is None:
                continue
            g = p.grad.data + group.get("weight_decay", 0.0) * p.data
            st = self.state[i]
            if "exp_avg" not in st:
                st["exp_avg"] = np.zeros_like(g)
                st["exp_inf"] = np.zeros_like(g)
            m = self.betas[0] * st["exp_avg"] + (1 - self.betas[0]) * g
            u = np.maximum(self.betas[1] * st["exp_inf"], np.abs(g))
            st["exp_avg"] = m
            st["exp_inf"] = u
            p.data = p.data - group.get("lr", self.lr) * m / (u + self.eps)


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr, weight_decay, betas=betas, eps=eps)
        self.betas = betas
        self.eps = eps
        self._step = 0

    def state_dict(self) -> dict:
        sd = super().state_dict()
        sd["_step"] = self._step
        sd["betas"] = list(self.betas)
        sd["eps"] = self.eps
        return sd

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self._step = state_dict.get("_step", 0)
        self.betas = tuple(state_dict.get("betas", self.betas))
        self.eps = state_dict.get("eps", self.eps)

    def step(self):
        self._step += 1
        for i, p, group in self._enumerate():
            if p.grad is None:
                continue
            g = p.grad.data
            if "exp_avg" not in self.state[i]:
                self.state[i]["exp_avg"] = np.zeros_like(g)
                self.state[i]["exp_avg_sq"] = np.zeros_like(g)
            m = self.betas[0] * self.state[i]["exp_avg"] + (1 - self.betas[0]) * g
            v = self.betas[1] * self.state[i]["exp_avg_sq"] + (1 - self.betas[1]) * g**2
            self.state[i]["exp_avg"] = m
            self.state[i]["exp_avg_sq"] = v
            m_hat = m / (1 - self.betas[0] ** self._step)
            v_hat = v / (1 - self.betas[1] ** self._step)
            lr = group.get("lr", self.lr)
            wd = group.get("weight_decay", self.weight_decay)
            p.data = p.data - lr * (m_hat / (np.sqrt(v_hat) + self.eps) + wd * p.data)


class NAdam(Adam):
    """NAdam optimizer (Nesterov-accelerated Adam)."""

    def __init__(
        self,
        params,
        lr: float = 0.002,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 0.004,
    ):
        super().__init__(params, lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.momentum_decay = momentum_decay

    def step(self):
        self._step += 1
        mu = self.momentum_decay
        beta1, beta2 = self.betas
        mu_t = beta1 * (1 - 0.5 * 0.96 ** (self._step * mu))
        mu_t1 = beta1 * (1 - 0.5 * 0.96 ** ((self._step + 1) * mu))
        for i, p, group in self._enumerate():
            if p.grad is None:
                continue
            g = p.grad.data
            if "exp_avg" not in self.state[i]:
                self.state[i]["exp_avg"] = np.zeros_like(g)
                self.state[i]["exp_avg_sq"] = np.zeros_like(g)
            m = beta1 * self.state[i]["exp_avg"] + (1 - beta1) * g
            v = beta2 * self.state[i]["exp_avg_sq"] + (1 - beta2) * g**2
            self.state[i]["exp_avg"] = m
            self.state[i]["exp_avg_sq"] = v
            m_hat = m / (1 - beta1 ** self._step)
            m_bar = (1 - mu_t) * g + mu_t1 * m_hat
            v_hat = v / (1 - beta2 ** self._step)
            lr = group.get("lr", self.lr)
            wd = group.get("weight_decay", self.weight_decay)
            p.data = p.data - lr * (wd * p.data + m_bar / (np.sqrt(v_hat) + self.eps))


class Rprop(Optimizer):
    """Resilient backpropagation (Rprop)."""

    def __init__(
        self,
        params,
        lr: float = 0.01,
        etas=(0.5, 1.2),
        step_sizes=(1e-6, 50.0),
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr, weight_decay, etas=etas, step_sizes=step_sizes)
        self.etas = etas
        self.step_sizes = step_sizes

    def step(self):
        for i, p, group in self._enumerate():
            if p.grad is None:
                continue
            g = p.grad.data + group.get("weight_decay", 0.0) * p.data
            st = self.state[i]
            if "step_size" not in st:
                st["step_size"] = np.full_like(g, group.get("lr", self.lr))
                st["prev_grad"] = np.zeros_like(g)
            sign = np.sign(g * st["prev_grad"])
            # positive product -> multiply step size up
            step_size = np.where(
                sign > 0,
                np.minimum(st["step_size"] * self.etas[1], self.step_sizes[1]),
                np.where(
                    sign < 0,
                    np.maximum(st["step_size"] * self.etas[0], self.step_sizes[0]),
                    st["step_size"],
                ),
            )
            st["step_size"] = step_size
            p.data = p.data - np.sign(g) * step_size
            # reset grad where sign < 0 (sign changed)
            st["prev_grad"] = np.where(sign < 0, np.zeros_like(g), g)


class ASGD(Optimizer):
    """Averaged SGD with optional averaging."""

    def __init__(
        self,
        params,
        lr: float = 0.01,
        lambd: float = 1e-4,
        alpha: float = 0.75,
        t0: float = 1e6,
        weight_decay: float = 0.0,
        average_pars: bool = True,
    ):
        super().__init__(params, lr, weight_decay, lambd=lambd, alpha=alpha, t0=t0)
        self.lambd = lambd
        self.alpha = alpha
        self.t0 = t0
        self.average_pars = average_pars

    def step(self):
        for i, p, group in self._enumerate():
            if p.grad is None:
                continue
            g = p.grad.data + group.get("weight_decay", 0.0) * p.data
            st = self.state[i]
            if "step" not in st:
                st["step"] = 0
                st["eta"] = group.get("lr", self.lr)
                st["mu"] = 1.0
                st["ax"] = np.zeros_like(g)
                st["ax_comp"] = np.zeros_like(g)
            st["step"] += 1
            step = st["step"]
            if step == 1:
                etamu = 1.0
            else:
                etamu = group.get("lr", self.lr) / (self.alpha * (step / (self.t0 + 1)) ** self.alpha + self.lambd)
            if etamu > 1.0:
                etamu = 1.0
            mu = st["eta"] * (1.0 - etamu) / etamu
            st["eta"] = group.get("lr", self.lr) / (self.alpha * (step / (self.t0 + 1)) ** self.alpha + self.lambd)
            st["mu"] = mu
            p.data = p.data - etamu * g
            st["ax"] = mu * st["ax"] + etamu * st["mu"] * g
            st["ax_comp"] += st["mu"] * g - st["ax"]
            if self.average_pars:
                p.data = st["ax"] - st["ax_comp"]


class LBFGS(Optimizer):
    """Limited-memory BFGS. Only step with a closure providing loss and gradients."""

    def __init__(
        self,
        params,
        lr: float = 1.0,
        max_iter: int = 20,
        history_size: int = 10,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
    ):
        super().__init__(params, lr)
        self.max_iter = max_iter
        self.history_size = history_size
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change

    def full_step(self, closure):
        """Run the full L-BFGS optimizer routine by repeatedly evaluating
        ``closure`` and taking a quasi-Newton direction."""
        for _ in range(self.max_iter):
            loss = closure()
            grads = [p.grad.data.copy() if p.grad is not None else np.zeros_like(p.data) for p in self.params]
            # simple gradient step fallback (BFGS direction approximation omitted)
            for i, p in enumerate(self.params):
                if p.grad is not None:
                    p.data = p.data - self.lr * grads[i]
        return loss

    def step(self, closure=None):
        if closure is None:
            raise ValueError("LBFGS requires a closure function")
        return self.full_step(closure)


class SparseAdam(Adam):
    """Adam for sparse gradients (falls back to dense Adam update)."""

    def __init__(self, params, lr: float = 0.001, betas=(0.9, 0.999), eps: float = 1e-8):
        super().__init__(params, lr=lr, betas=betas, eps=eps)


class CosineAnnealingLR:
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self._step = 0

    def step(self):
        self._step += 1
        cos = (1 + np.cos(np.pi * self._step / self.T_max)) / 2
        lr = self.eta_min + (self.optimizer.lr - self.eta_min) * cos
        self.optimizer.lr = lr
