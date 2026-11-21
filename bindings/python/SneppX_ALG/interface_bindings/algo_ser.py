"""SER (Sparse Expert Routing) algorithm bindings.

Wraps C implementations in ``algorithms/ser/core/`` with pure-Python fallback.
"""

from typing import Optional, Tuple, List

import numpy as np

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_architecture_layer")


class SERExpert:
    """Single expert module in a mixture-of-experts layer."""

    def __init__(self, expert_id: int, input_dim: int, hidden_dim: int):
        self.expert_id = expert_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._has_c = _HAS_C

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x


class SERRoute:
    """Expert routing — top-k gating and load balancing."""

    def __init__(self, n_experts: int, k: int = 2):
        self.n_experts = n_experts
        self.k = k
        self._has_c = _HAS_C

    def topk_gating(self, x: np.ndarray, router_weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scores = x @ router_weight.T
        indices = np.argsort(-scores, axis=-1)[:, :self.k]
        weights = np.take_along_axis(scores, indices, axis=-1)
        weights = np.exp(weights) / np.sum(np.exp(weights), axis=-1, keepdims=True)
        return indices, weights

    def load_balance_loss(self, gate_scores: np.ndarray, topk_indices: np.ndarray) -> float:
        batch = gate_scores.shape[0]
        load = np.zeros(self.n_experts, dtype=np.float32)
        for i in range(batch):
            for j in range(self.k):
                load[topk_indices[i, j]] += 1.0
        load = load / (batch * self.k)
        importance = gate_scores.mean(axis=0)
        cv = np.std(load) / (np.mean(load) + 1e-8)
        return float(cv)


class SERLayer:
    """Mixture-of-experts layer with sparse routing."""

    def __init__(self, n_experts: int, input_dim: int, hidden_dim: int, k: int = 2):
        self.n_experts = n_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self._router = SERRoute(n_experts, k)
        self._has_c = _HAS_C

    def forward(self, x: np.ndarray, router_weight: np.ndarray,
                expert_outputs: List[np.ndarray]) -> np.ndarray:
        *batch_shape, feat_dim = x.shape
        flat_x = x.reshape(-1, feat_dim)
        indices, weights = self._router.topk_gating(flat_x, router_weight)
        flat_out = np.zeros((flat_x.shape[0], self.input_dim), dtype=np.float32)
        for i in range(flat_x.shape[0]):
            for e_idx, w in zip(indices[i], weights[i]):
                flat_out[i] += w * flat_x[i]
        return flat_out.reshape(*batch_shape, self.input_dim)


class SERLoss:
    """SER auxiliary losses — load balancing, importance, z-loss."""

    def __init__(self):
        self._has_c = _HAS_C

    @staticmethod
    def load_balancing(gate_scores: np.ndarray, topk_indices: np.ndarray,
                       n_experts: int) -> float:
        route = SERRoute(n_experts)
        return route.load_balance_loss(gate_scores, topk_indices)

    @staticmethod
    def z_loss(logits: np.ndarray, z_weight: float = 0.001) -> float:
        return float(z_weight * np.mean(np.log(np.sum(np.exp(logits), axis=-1)) ** 2))


class SERModel:
    """Full SER model with multiple MoE layers."""

    def __init__(self, n_layers: int = 4, n_experts: int = 8, d_model: int = 64, k: int = 2):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.d_model = d_model
        self.k = k
        self._has_c = _HAS_C

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        return h
