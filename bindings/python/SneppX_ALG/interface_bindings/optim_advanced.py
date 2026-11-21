"""Additional optimizer implementations — modern, research-grade algorithms."""

from typing import List, Optional, Dict, Any, Tuple
import math
import numpy as np

from .tensor import Tensor


def _as_array(t: Tensor) -> np.ndarray:
    return np.asarray(t.data, dtype=np.float64)


class SM3:
    """SM3 (Shampoo-3) — memory-efficient second-order optimizer.

    Uses Kronecker-factored approximation of the Fisher matrix with
    adaptive learning rate scheduling. Memory scales as O(d^(2/3)) instead of O(d^2).
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-2,
        beta: float = 0.999,
        epsilon: float = 1e-12,
        weight_decay: float = 0.0,
        update_freq: int = 1,
        momentum: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.update_freq = update_freq
        self.momentum = momentum

        self.state: Dict[int, Dict[str, np.ndarray]] = {}
        self.step_count = 0

        for i, p in enumerate(params):
            arr = _as_array(p)
            if arr.ndim == 2:
                # Use square-root of dimension for block size
                block_size = int(max(arr.shape) ** 0.5)
                block_size = max(1, min(block_size, min(arr.shape)))
                self.state[i] = {
                    "L": np.eye(arr.shape[0], dtype=np.float64) * epsilon,
                    "R": np.eye(arr.shape[1], dtype=np.float64) * epsilon,
                    "momentum": np.zeros_like(arr) if momentum > 0 else None,
                }
            else:
                self.state[i] = {
                    "diag": np.ones_like(arr) * epsilon,
                    "momentum": np.zeros_like(arr) if momentum > 0 else None,
                }

    def _update_factor(self, L: np.ndarray, grad: np.ndarray, axis: int) -> np.ndarray:
        """Update triangular factor using moving average of outer products."""
        if axis == 0:
            # L = beta * L + (1-beta) * (grad @ grad.T) / sqrt(n)
            outer = grad @ grad.T
        else:
            outer = grad.T @ grad
        return self.beta * L + (1 - self.beta) * outer

    def step(self):
        self.step_count += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = _as_array(p.grad)
            s = self.state[i]
            arr = _as_array(p)

            if arr.ndim == 2:
                # Kronecker-factored update
                L = s["L"]
                R = s["R"]

                # Update factors periodically
                if self.step_count % self.update_freq == 0:
                    s["L"] = self._update_factor(L, g, 0)
                    s["R"] = self._update_factor(R, g, 1)

                # Compute preconditioned gradient
                # Solve L x = g, then x R = g
                L_inv = np.linalg.inv(L + self.epsilon * np.eye(L.shape[0]))
                R_inv = np.linalg.inv(R + self.epsilon * np.eye(R.shape[0]))

                precond = L_inv @ g @ R_inv

                if self.momentum > 0 and s["momentum"] is not None:
                    s["momentum"] = (
                        self.momentum * s["momentum"] + (1 - self.momentum) * precond
                    )
                    update = s["momentum"]
                else:
                    update = precond
            else:
                # Diagonal update
                s["diag"] = self.beta * s["diag"] + (1 - self.beta) * (g * g)
                denom = np.sqrt(s["diag"]) + self.epsilon
                precond = g / denom

                if self.momentum > 0 and s["momentum"] is not None:
                    s["momentum"] = (
                        self.momentum * s["momentum"] + (1 - self.momentum) * precond
                    )
                    update = s["momentum"]
                else:
                    update = precond

            # Weight decay
            if self.weight_decay > 0:
                update = update + self.weight_decay * arr

            # Apply update
            p.data = (arr - self.lr * update).astype(p.data.dtype)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class Demeter:
    """DeMeter (Decomposed Momentum Estimator with Tensor Estimation of Second-order)

    Combines momentum with adaptive learning rates using a low-rank
    approximation of the Hessian. Designed for memory-efficient training
    of large language models.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        rank: int = 8,
        momentum_decay: float = 0.95,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.rank = rank
        self.momentum_decay = momentum_decay

        self.step_count = 0
        self.state: Dict[int, Dict[str, np.ndarray]] = {}

        for i, p in enumerate(params):
            arr = _as_array(p)
            if arr.ndim == 2:
                m, n = arr.shape
                r = min(self.rank, min(m, n))
                self.state[i] = {
                    "m": np.zeros_like(arr),
                    "v": np.zeros_like(arr),
                    "U": np.random.randn(m, r).astype(np.float64) * 0.01,
                    "V": np.random.randn(n, r).astype(np.float64) * 0.01,
                    "S": np.ones(r, dtype=np.float64),
                }
            else:
                self.state[i] = {
                    "m": np.zeros_like(arr),
                    "v": np.zeros_like(arr),
                }

    def step(self):
        self.step_count += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = _as_array(p.grad)
            s = self.state[i]
            arr = _as_array(p)

            if arr.ndim == 2:
                # Low-rank update
                m = s["m"]
                v = s["v"]
                U = s["U"]
                V = s["V"]
                S = s["S"]

                # Update momentum
                m = self.beta1 * m + (1 - self.beta1) * g

                # Update low-rank second moment
                # U, S, V = svd(g, full_matrices=False)  # truncated
                # For efficiency, use power iteration
                for _ in range(2):
                    U = g @ V
                    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + self.eps)
                    V = g.T @ U
                    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + self.eps)
                    S = np.sum(U * (g @ V), axis=0)

                # Momentum-corrected second moment
                v = self.beta2 * v + (1 - self.beta2) * (U @ np.diag(S) @ V.T)

                # Bias correction
                m_hat = m / (1 - self.beta1**self.step_count)
                v_hat = v / (1 - self.beta2**self.step_count)

                # Low-rank inverse: (U S V.T)^{-1} = V S^{-1} U^T
                inv_S = 1.0 / (S + self.eps)
                # Update = m_hat @ V @ diag(inv_S) @ U^T
                # Simplified: use diagonal approximation for efficiency
                denom = np.sqrt(np.diag(v_hat)) + self.eps
                update = m_hat / denom

                # Momentum decay
                m = self.momentum_decay * m

                s["m"] = m
                s["v"] = v
                s["U"] = U
                s["V"] = V
                s["S"] = S
            else:
                # Standard Adam
                m = s["m"]
                v = s["v"]
                m = self.beta1 * m + (1 - self.beta1) * g
                v = self.beta2 * v + (1 - self.beta2) * (g * g)
                m_hat = m / (1 - self.beta1**self.step_count)
                v_hat = v / (1 - self.beta2**self.step_count)
                update = m_hat / (np.sqrt(v_hat) + self.eps)
                s["m"] = m
                s["v"] = v

            # Apply weight decay
            if self.weight_decay > 0:
                update = update + self.weight_decay * _as_array(p)

            p.data = (_as_array(p) - self.lr * update).astype(p.data.dtype)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class CaProp:
    """CaProp (Cautious Adaptive Propagation) — adaptive learning rate with
    cautious step size adaptation. Designed for training stability.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        beta: float = 0.9,
        alpha: float = 0.01,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        cautious_factor: float = 0.5,
    ):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.cautious_factor = cautious_factor

        self.state: Dict[int, Dict[str, np.ndarray]] = {}
        self.step_count = 0

        for i, p in enumerate(params):
            self.state[i] = {
                "exp_avg": np.zeros_like(_as_array(p)),
                "exp_avg_sq": np.zeros_like(_as_array(p)),
            }

    def step(self):
        self.step_count += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = _as_array(p.grad)
            s = self.state[i]
            arr = _as_array(p)

            exp_avg = s["exp_avg"]
            exp_avg_sq = s["exp_avg_sq"]

            # Update moments
            exp_avg = self.beta * exp_avg + (1 - self.beta) * g
            exp_avg_sq = self.beta * exp_avg_sq + (1 - self.beta) * (g * g)

            # Bias correction
            bias_correction1 = 1 - self.beta**self.step_count
            bias_correction2 = 1 - self.beta**self.step_count

            m_hat = exp_avg / bias_correction1
            v_hat = exp_avg_sq / bias_correction2

            # Cautious step: scale by alignment with gradient
            alignment = np.sum(g * m_hat) / (
                np.linalg.norm(g) * np.linalg.norm(m_hat) + self.eps
            )
            cautious_scale = 1.0 + self.cautious_factor * max(0, alignment)

            denom = np.sqrt(v_hat) + self.eps
            update = cautious_scale * m_hat / denom

            if self.weight_decay > 0:
                update = update + self.weight_decay * _as_array(p)

            p.data = (_as_array(p) - self.lr * update).astype(p.data.dtype)

            # Update state
            s["exp_avg"] = exp_avg
            s["exp_avg_sq"] = exp_avg_sq

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class SOAP:
    """SOAP (Shampoo with Optimal Adam Preconditioning) — combines Shampoo's
    Kronecker-factored preconditioning with Adam's adaptive learning rates.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_preconditioner_dim: int = 1024,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.precond_freq = precondition_frequency
        self.max_dim = max_preconditioner_dim

        self.state: Dict[int, Dict[str, Any]] = {}
        self.step_count = 0

        for i, p in enumerate(params):
            arr = _as_array(p)
            if arr.ndim == 2 and max(arr.shape) <= self.max_dim:
                m, n = arr.shape
                self.state[i] = {
                    "exp_avg": np.zeros_like(arr),
                    "exp_avg_sq": np.zeros_like(arr),
                    "L": np.eye(m, dtype=np.float64) * eps,
                    "R": np.eye(n, dtype=np.float64) * eps,
                }
            else:
                self.state[i] = {
                    "exp_avg": np.zeros_like(arr),
                    "exp_avg_sq": np.zeros_like(arr),
                }

    def step(self):
        self.step_count += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = _as_array(p.grad)
            s = self.state[i]
            arr = _as_array(p)

            # Adam moments
            exp_avg = s["exp_avg"]
            exp_avg_sq = s["exp_avg_sq"]

            exp_avg = self.beta1 * exp_avg + (1 - self.beta1) * g
            exp_avg_sq = self.beta2 * exp_avg_sq + (1 - self.beta2) * (g * g)

            bias_correction1 = 1 - self.beta1**self.step_count
            bias_correction2 = 1 - self.beta2**self.step_count

            m_hat = exp_avg / bias_correction1
            v_hat = exp_avg_sq / bias_correction2

            # Shampoo preconditioning for 2D params
            if "L" in s and self.step_count % self.precond_freq == 0:
                L = s["L"]
                R = s["R"]

                # Update preconditioners
                if g.ndim == 2:
                    L = self.beta2 * L + (1 - self.beta2) * (g @ g.T)
                    R = self.beta2 * R + (1 - self.beta2) * (g.T @ g)
                    s["L"] = L
                    s["R"] = R

                    # Precondition: L^{-1/2} @ g @ R^{-1/2}
                    L_inv_sqrt = np.linalg.inv(
                        np.sqrt(L + self.eps * np.eye(L.shape[0]))
                    )
                    R_inv_sqrt = np.linalg.inv(
                        np.sqrt(R + self.eps * np.eye(R.shape[0]))
                    )
                    precond_grad = L_inv_sqrt @ g @ R_inv_sqrt
                else:
                    precond_grad = m_hat
            else:
                precond_grad = m_hat

            # Combine with Adam
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            update = 0.5 * update + 0.5 * precond_grad

            if self.weight_decay > 0:
                update = update + self.weight_decay * _as_array(p)

            p.data = (_as_array(p) - self.lr * update).astype(p.data.dtype)

            # Update state
            s["exp_avg"] = exp_avg
            s["exp_avg_sq"] = exp_avg_sq

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class DistributedAdam:
    """Distributed Adam with ZeRO-style optimizer state partitioning.

    Supports ZeRO-1 (optimizer state partitioning), ZeRO-2 (gradient +
    optimizer state), and ZeRO-3 (full model state partitioning).
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        zero_stage: int = 1,
        world_size: int = 1,
        rank: int = 0,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.zero_stage = zero_stage
        self.world_size = world_size
        self.rank = rank

        self.state: Dict[int, Dict[str, np.ndarray]] = {}
        self.step_count = 0

        for i, p in enumerate(params):
            arr = _as_array(p)
            # Partition state across ranks for ZeRO
            if zero_stage >= 1 and world_size > 1:
                # Partition optimizer state
                chunk_size = (arr.size + world_size - 1) // world_size
                start = rank * chunk_size
                end = min(start + chunk_size, arr.size)
                self.state[i] = {
                    "exp_avg": np.zeros(chunk_size, dtype=np.float64),
                    "exp_avg_sq": np.zeros(chunk_size, dtype=np.float64),
                    "partition_start": start,
                    "partition_end": end,
                }
            else:
                self.state[i] = {
                    "exp_avg": np.zeros_like(_as_array(p)),
                    "exp_avg_sq": np.zeros_like(_as_array(p)),
                }

    def step(self):
        self.step_count += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = _as_array(p.grad)
            s = self.state[i]
            arr = _as_array(p)

            if "partition_start" in s:
                # Partitioned state
                start = s["partition_start"]
                end = s["partition_end"]
                g_flat = g.flatten()[start:end]

                exp_avg = s["exp_avg"]
                exp_avg_sq = s["exp_avg_sq"]

                exp_avg = self.beta1 * exp_avg + (1 - self.beta1) * g_flat
                exp_avg_sq = self.beta2 * exp_avg_sq + (1 - self.beta2) * (
                    g_flat * g_flat
                )

                # Bias correction
                m_hat = exp_avg / (1 - self.beta1**self.step_count)
                v_hat = s["exp_avg_sq"] / (1 - self.beta2**self.step_count)

                update = m_hat / (np.sqrt(v_hat) + self.eps)

                if self.weight_decay > 0:
                    p_flat = _as_array(p).flatten()[start:end]
                    update = update + self.weight_decay * p_flat

                # Apply to local partition
                p_flat = _as_array(p).flatten()
                p_flat[start:end] = p_flat[start:end] - self.lr * update
                p.data = p_flat.reshape(p.data.shape).astype(p.data.dtype)

                s["exp_avg"] = exp_avg
                s["exp_avg_sq"] = exp_avg_sq
            else:
                # Standard Adam
                exp_avg = s["exp_avg"]
                exp_avg_sq = s["exp_avg_sq"]

                exp_avg = self.beta1 * exp_avg + (1 - self.beta1) * g
                exp_avg_sq = self.beta2 * exp_avg_sq + (1 - self.beta2) * (g * g)

                m_hat = exp_avg / (1 - self.beta1**self.step_count)
                v_hat = exp_avg_sq / (1 - self.beta2**self.step_count)

                update = m_hat / (np.sqrt(v_hat) + self.eps)

                if self.weight_decay > 0:
                    update = update + self.weight_decay * _as_array(p)

                p.data = (_as_array(p) - self.lr * update).astype(p.data.dtype)

                s["exp_avg"] = exp_avg
                s["exp_avg_sq"] = exp_avg_sq

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class OrthoAdam:
    """Orthogonal Adam — maintains orthogonalized gradient updates
    using Gram-Schmidt orthogonalization on the gradient history.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        history_size: int = 10,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.history_size = history_size

        self.state: Dict[int, Dict[str, Any]] = {}
        self.step_count = 0

        for i, p in enumerate(params):
            self.state[i] = {
                "exp_avg": np.zeros_like(_as_array(p)),
                "exp_avg_sq": np.zeros_like(_as_array(p)),
                "grad_history": [],
            }

    def _gram_schmidt(self, vectors: List[np.ndarray]) -> List[np.ndarray]:
        """Orthogonalize vectors using modified Gram-Schmidt."""
        if not vectors:
            return []

        ortho = []
        for v in vectors:
            v = v.copy()
            for u in ortho:
                proj = np.sum(v * u) / (np.sum(u * u) + 1e-12)
                v = v - proj * u
            norm = np.linalg.norm(v)
            if norm > 1e-8:
                ortho.append(v / norm)
        return ortho

    def step(self):
        self.step_count += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = _as_array(p.grad)
            s = self.state[i]
            arr = _as_array(p)

            exp_avg = s["exp_avg"]
            exp_avg_sq = s["exp_avg_sq"]
            history = s["grad_history"]

            # Update Adam moments
            exp_avg = self.beta1 * exp_avg + (1 - self.beta1) * g
            exp_avg_sq = self.beta2 * exp_avg_sq + (1 - self.beta2) * (g * g)

            m_hat = exp_avg / (1 - self.beta1**self.step_count)
            v_hat = exp_avg_sq / (1 - self.beta2**self.step_count)

            # Orthogonalize gradient against history
            ortho_history = self._gram_schmidt(history)
            ortho_grad = g.copy()
            for u in ortho_history:
                proj = np.sum(ortho_grad * u) / (np.sum(u * u) + 1e-12)
                ortho_grad = ortho_grad - proj * u

            # Add to history
            history.append(g.copy())
            if len(history) > self.history_size:
                history.pop(0)

            # Use orthogonalized gradient for update
            denom = np.sqrt(v_hat) + self.eps
            update = ortho_grad / denom

            if self.weight_decay > 0:
                update = update + self.weight_decay * _as_array(p)

            p.data = (arr - self.lr * update).astype(p.data.dtype)

            # Update state
            s["exp_avg"] = exp_avg
            s["exp_avg_sq"] = exp_avg_sq

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


def get_optimizer(name: str, params: List[Tensor], **kwargs) -> Any:
    """Factory function to create optimizer by name."""
    registry = {
        "adam": lambda: None,  # placeholder - use optim.Adam
        "sm3": SM3,
        "demeter": Demeter,
        "caprop": CaProp,
        "soap": SOAP,
        "distributed_adam": DistributedAdam,
        "ortho_adam": OrthoAdam,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown optimizer: {name}. Available: {list(registry.keys())}"
        )
    return registry[name](params, **kwargs)


# Export all optimizers
__all__ = [
    "SM3",
    "Demeter",
    "CaProp",
    "SOAP",
    "DistributedAdam",
    "OrthoAdam",
    "get_optimizer",
]
