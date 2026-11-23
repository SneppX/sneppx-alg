"""Differential Privacy — Laplace/Gaussian mechanisms, RDP accountant, DP-SGD."""

import math
import os
import json
import numpy as np
from typing import Optional, List, Tuple, Dict, Union


# ===========================================================================
#  Noise Mechanisms
# ===========================================================================


def _secure_uniform(shape) -> np.ndarray:
    """Uniform samples in [0, 1) drawn directly from OS entropy (CSPRNG)."""
    count = int(np.prod(shape)) if shape else 1
    words = np.frombuffer(os.urandom(8 * max(count, 1)), dtype=np.uint64)
    return (words >> 11).astype(np.float64).reshape(shape) / float(1 << 53)


def _secure_laplace(scale: float, shape) -> np.ndarray:
    u = _secure_uniform(shape) - 0.5
    u = np.where(u == 0.0, 0.5 / float(1 << 53), u)
    return -scale * np.sign(u) * np.log1p(-2.0 * np.abs(u))


def _secure_normal(sigma: float, shape) -> np.ndarray:
    u1 = np.maximum(_secure_uniform(shape), 0.5 / float(1 << 53))
    u2 = _secure_uniform(shape)
    return sigma * np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)


class LaplaceMech:
    """epsilon-DP Laplace mechanism.

    Adds noise drawn from Lap(0, sensitivity/epsilon) to a value.
    """

    def __init__(self, epsilon: float):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self.epsilon = epsilon

    def apply(
        self, value: Union[float, np.ndarray], sensitivity: float = 1.0
    ) -> np.ndarray:
        scale = sensitivity / self.epsilon
        noise = _secure_laplace(scale, np.shape(value))
        return np.asarray(value, dtype=np.float64) + noise


class GaussianMech:
    """(epsilon, delta)-DP Gaussian mechanism.

    Adds noise drawn from N(0, sigma^2) where:
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
    """

    def __init__(self, epsilon: float, delta: float = 1e-5):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.epsilon = epsilon
        self.delta = delta

    def apply(
        self, value: Union[float, np.ndarray], sensitivity: float = 1.0
    ) -> np.ndarray:
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        noise = _secure_normal(sigma, np.shape(value))
        return np.asarray(value, dtype=np.float64) + noise


# ===========================================================================
#  Privacy Budget (sequential composition)
# ===========================================================================


class PrivacyBudget:
    """Track and enforce privacy budget via sequential composition.

    ``epsilon`` and ``delta`` are consumed additively.  Raises when
    budget would be exceeded.
    """

    def __init__(self, epsilon: float, delta: float = 1e-5):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._total_epsilon = epsilon
        self._total_delta = delta
        self._spent_epsilon = 0.0
        self._spent_delta = 0.0

    def check(self, epsilon: float, delta: float = 0.0) -> bool:
        return (
            self._spent_epsilon + epsilon <= self._total_epsilon
            and self._spent_delta + delta <= self._total_delta
        )

    def spend(self, epsilon: float, delta: float = 0.0):
        if not self.check(epsilon, delta):
            raise ValueError(
                f"Privacy budget exhausted: "
                f"epsilon {self._spent_epsilon + epsilon:.4f} > {self._total_epsilon:.4f} "
                f"or delta {self._spent_delta + delta:.6e} > {self._total_delta:.6e}"
            )
        self._spent_epsilon += epsilon
        self._spent_delta += delta

    @property
    def remaining(self) -> Tuple[float, float]:
        return (
            self._total_epsilon - self._spent_epsilon,
            self._total_delta - self._spent_delta,
        )

    @property
    def spent(self) -> Tuple[float, float]:
        return (self._spent_epsilon, self._spent_delta)

    def to_dict(self) -> dict:
        return {
            "total_epsilon": self._total_epsilon,
            "total_delta": self._total_delta,
            "spent_epsilon": self._spent_epsilon,
            "spent_delta": self._spent_delta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PrivacyBudget":
        budget = cls(d["total_epsilon"], d["total_delta"])
        budget._spent_epsilon = d["spent_epsilon"]
        budget._spent_delta = d["spent_delta"]
        return budget


# ===========================================================================
#  Rényi Differential Privacy (RDP) Accountant
# ===========================================================================


class RDPAccountant:
    """Rényi Differential Privacy accountant.

    Tracks cumulative RDP across training steps and converts to
    (epsilon, delta)-DP at query time.

    Supports:
    - Gaussian mechanism (per-step RDP = alpha / (2 * sigma^2))
    - Poisson subsampling amplification (approximate: rdp *= q^2)
    - RDP -> (epsilon, delta) conversion via optimal order selection
    """

    def __init__(
        self,
        delta: float = 1e-5,
        orders: Optional[List[float]] = None,
    ):
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self._delta = delta
        self._orders = (
            orders
            if orders is not None
            else [1 + 0.1 * i for i in range(1, 100)]
            + list(range(11, 65))
            + [128, 256, 512]
        )
        self._rdp = np.zeros(len(self._orders))
        self._steps = 0

    def step(
        self,
        noise_multiplier: float,
        batch_size: int,
        num_samples: Optional[int] = None,
    ):
        """Account for one training step.

        Args:
            noise_multiplier: sigma / max_grad_norm ratio.
            batch_size: number of samples in this step.
            num_samples: total dataset size (needed for amplification).
        """
        if noise_multiplier <= 0:
            return

        q = batch_size / num_samples if (num_samples and num_samples > 0) else 1.0

        for i, alpha in enumerate(self._orders):
            if alpha <= 1:
                continue
            # RDP for Gaussian: alpha / (2 * sigma^2)
            # sigma = noise_multiplier * 1.0 (sensitivity = 1 for clipped gradient)
            step_rdp = alpha / (2.0 * noise_multiplier * noise_multiplier)

            # Subsampling amplification (Poisson, dominant-term approx)
            if q < 1.0:
                # E[ (q * N(0,sigma^2) + (1-q))^alpha ] approximation
                step_rdp = math.log(
                    1.0
                    + q
                    * q
                    * alpha
                    * (alpha - 1)
                    / (2.0 * noise_multiplier * noise_multiplier)
                ) / (alpha - 1)
                if step_rdp < 0:
                    step_rdp = 0.0

            self._rdp[i] += step_rdp

        self._steps += 1

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """Convert cumulative RDP to (epsilon, delta)-DP.

        Uses optimal order selection:
            epsilon(alpha) = rdp(alpha) + log(1/delta) / (alpha - 1)
        Returns min over alpha > 1.
        Returns ``inf`` if no steps have been accounted.
        """
        if self._steps == 0:
            return float("inf")
        target_delta = delta if delta is not None else self._delta
        epsilons = []
        for i, alpha in enumerate(self._orders):
            if alpha <= 1:
                continue
            eps = self._rdp[i] + math.log(1.0 / target_delta) / (alpha - 1)
            epsilons.append(eps)
        return float(min(epsilons)) if epsilons else float("inf")

    def to_dict(self) -> dict:
        return {
            "delta": self._delta,
            "orders": self._orders,
            "rdp": self._rdp.tolist(),
            "steps": self._steps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RDPAccountant":
        acc = cls(delta=d["delta"], orders=d["orders"])
        acc._rdp = np.array(d["rdp"])
        acc._steps = d["steps"]
        return acc


# ===========================================================================
#  DP-SGD Optimizer Wrapper
# ===========================================================================


class DPSGD:
    """DP-SGD that wraps a base optimizer.

    For each step:
    1. Compute per-sample gradients (one backward per sample).
    2. Clip each per-sample gradient to ``max_grad_norm``.
    3. Average clipped gradients across the batch.
    4. Add Gaussian noise: sigma = noise_multiplier * max_grad_norm / batch_size.
    5. Call wrapped optimizer's ``step()``.

    This replaces the normal ``loss.backward()`` + ``optimizer.step()`` flow.
    Expects a **list of per-sample losses** as input to ``step()``.
    """

    def __init__(
        self,
        optimizer,
        noise_multiplier: float,
        max_grad_norm: float = 1.0,
        num_samples: Optional[int] = None,
        accountant: Optional[RDPAccountant] = None,
        epsilon: Optional[float] = None,
        delta: float = 1e-5,
        budget: Optional[PrivacyBudget] = None,
    ):
        self._opt = optimizer
        self._params = optimizer.params
        self._noise_mult = noise_multiplier
        self._max_grad_norm = max_grad_norm
        self._num_samples = num_samples
        self._steps = 0
        self._epsilon = epsilon
        self._delta = delta

        if accountant is not None:
            self._accountant = accountant
        elif epsilon is not None:
            self._accountant = RDPAccountant(delta=delta)
        else:
            self._accountant = None

        # M-7: accept epsilon?/delta? as an ENFORCED budget, not decoration.
        self._budget = budget
        if self._budget is None and epsilon is not None:
            self._budget = PrivacyBudget(epsilon, delta)

    @property
    def lr(self):
        return self._opt.lr

    @lr.setter
    def lr(self, v):
        self._opt.lr = v

    @property
    def params(self):
        return self._params

    def zero_grad(self):
        self._opt.zero_grad()

    def _check_budget(self, batch_size: int):
        """Reject a step that would exceed the configured epsilon budget."""
        if self._epsilon is None or self._accountant is None or self._noise_mult <= 0:
            return
        import copy

        trial = copy.deepcopy(self._accountant)
        trial.step(self._noise_mult, batch_size, self._num_samples)
        projected = trial.get_epsilon(self._delta)
        if projected > self._epsilon:
            raise ValueError(
                f"epsilon budget exhausted: step would spend {projected:.4f} > {self._epsilon:.4f}"
            )

    def step(self, per_sample_losses):
        """Take a DP-SGD step using per-sample losses.

        Args:
            per_sample_losses: list of ``Tensor``, one scalar per sample.
        """
        batch_size = len(per_sample_losses)
        if batch_size == 0:
            return
        self._check_budget(batch_size)

        # M-7: refuse to spend past an exhausted privacy budget.
        if self._budget is not None and self._noise_mult > 0:
            if self._budget.remaining[0] <= 0:
                raise ValueError(
                    "Privacy budget exhausted: refusing additional DP-SGD step"
                )

        # Accumulate clipped per-sample gradients
        accumulated = [np.zeros_like(p.data, dtype=np.float64) for p in self._params]

        for loss in per_sample_losses:
            self._opt.zero_grad()
            loss.backward()

            per_sample_grads = []
            for p in self._params:
                if p.grad is not None:
                    per_sample_grads.append(p.grad.data.astype(np.float64).copy())
                else:
                    per_sample_grads.append(np.zeros_like(p.data, dtype=np.float64))

            total_norm_sq = sum(np.sum(g**2) for g in per_sample_grads)
            total_norm = float(np.sqrt(total_norm_sq))

            clip_coef = self._max_grad_norm / (total_norm + 1e-8)
            if clip_coef < 1.0:
                for i, g in enumerate(per_sample_grads):
                    accumulated[i] += g * clip_coef
            else:
                for i, g in enumerate(per_sample_grads):
                    accumulated[i] += g

        for i in range(len(self._params)):
            accumulated[i] /= batch_size

        sigma = self._noise_mult * self._max_grad_norm / batch_size

        from .tensor import Tensor

        for i in range(len(self._params)):
            noise = _secure_normal(sigma, self._params[i].data.shape)
            accumulated[i] += noise
            # Note: tensor.data returns a copy — must create a new Tensor to set grad
            self._params[i].grad = Tensor(
                accumulated[i].astype(self._params[i].data.dtype)
            )

        self._opt.step()

        self._steps += 1
        if self._accountant is not None:
            self._accountant.step(
                noise_multiplier=self._noise_mult,
                batch_size=batch_size,
                num_samples=self._num_samples,
            )
            if self._budget is not None and self._noise_mult > 0:
                # RDP->DP conversion has a delta term and optimal-alpha
                # selection, so epsilon is NOT additive; track the accountant's
                # actual cumulative value rather than summing step deltas.
                epsilon_after = self._accountant.get_epsilon()
                self._budget._spent_epsilon = epsilon_after
                if epsilon_after > self._budget._total_epsilon:
                    raise ValueError(
                        "Privacy budget exhausted: "
                        f"epsilon {epsilon_after:.4f} > {self._budget._total_epsilon:.4f}"
                    )

    def privacy_spent(self) -> Dict[str, float]:
        """Return (epsilon, delta) privacy cost so far."""
        if self._accountant is not None:
            return {
                "epsilon": self._accountant.get_epsilon(),
                "delta": self._accountant._delta,
            }
        return {"epsilon": float("inf"), "delta": 0.0}

    def state_dict(self) -> dict:
        return {
            "noise_multiplier": self._noise_mult,
            "max_grad_norm": self._max_grad_norm,
            "steps": self._steps,
            "num_samples": self._num_samples,
            "accountant": self._accountant.to_dict() if self._accountant else None,
            "opt_state": self._opt.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        self._noise_mult = state_dict.get("noise_multiplier", self._noise_mult)
        self._max_grad_norm = state_dict.get("max_grad_norm", self._max_grad_norm)
        self._steps = state_dict.get("steps", 0)
        self._num_samples = state_dict.get("num_samples", self._num_samples)
        if self._accountant and state_dict.get("accountant"):
            self._accountant = RDPAccountant.from_dict(state_dict["accountant"])
        if state_dict.get("opt_state"):
            self._opt.load_state_dict(state_dict["opt_state"])


__all__ = [
    "LaplaceMech",
    "GaussianMech",
    "PrivacyBudget",
    "RDPAccountant",
    "DPSGD",
]
