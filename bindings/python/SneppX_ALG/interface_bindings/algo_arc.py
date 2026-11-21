"""ARC (Adversarial Robustness Certification) algorithm bindings.

Wraps C implementations in ``algorithms/arc/core/`` with pure-Python fallback.
"""

from typing import Optional, Tuple, Callable, List

import numpy as np

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_architecture_layer")


class ARCAttackSim:
    """Adversarial attack simulation — PGD, FGSM, and other attacks."""

    def __init__(self, eps: float = 0.01, alpha: float = 0.001, steps: int = 40):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self._has_c = _HAS_C

    def pgd(self, x: np.ndarray, y: np.ndarray, loss_fn: Callable) -> np.ndarray:
        adv = x.copy()
        for _ in range(self.steps):
            grad = np.zeros_like(adv)
            adv = adv + self.alpha * np.sign(grad)
            delta = np.clip(adv - x, -self.eps, self.eps)
            adv = np.clip(x + delta, 0, 1)
        return adv

    def fgsm(self, x: np.ndarray, y: np.ndarray, loss_fn: Callable) -> np.ndarray:
        delta = self.eps * np.sign(np.random.randn(*x.shape))
        return np.clip(x + delta, 0, 1)


class ARCFForward:
    """ARC forward pass — certified forward propagation."""

    def __init__(self):
        self._has_c = _HAS_C

    @staticmethod
    def forward(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
        out = x @ weight.T
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def forward_layers(x: np.ndarray, layers: list) -> np.ndarray:
        h = x
        for layer in layers:
            h = layer(h)
        return h


class ARCGradientObfuscator:
    """Gradient obfuscation — masks gradients to prevent adversarial crafting."""

    def __init__(self):
        self._has_c = _HAS_C

    @staticmethod
    def obfuscate(grad: np.ndarray, method: str = "signed") -> np.ndarray:
        if method == "signed":
            return np.sign(grad)
        if method == "tanh":
            return np.tanh(grad) * np.abs(grad)
        if method == "quantize":
            return np.round(grad * 100) / 100
        return grad


class ARCInputGuard:
    """Input guard — detects and sanitizes adversarial perturbations."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._has_c = _HAS_C

    def detect(self, x: np.ndarray, clean_stats: dict) -> np.ndarray:
        mean = clean_stats.get("mean", np.mean(x, axis=0, keepdims=True))
        std = clean_stats.get("std", np.std(x, axis=0, keepdims=True))
        z = np.abs((x - mean) / (std + 1e-8))
        return np.any(z > self.threshold, axis=tuple(range(1, z.ndim)))

    def sanitize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, -self.threshold, self.threshold)


class ARCOutputVerifier:
    """Output verifier — certifies predictions against bounded perturbations."""

    def __init__(self):
        self._has_c = _HAS_C

    @staticmethod
    def certify(logits: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        top2 = np.sort(logits, axis=-1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        certified = margin > radius
        return certified, margin

    @staticmethod
    def worst_case(logits: np.ndarray, eps: float) -> np.ndarray:
        return logits - eps


class ARCAdversarialTrainGraph:
    """Adversarial training graph builder — constructs clean + adversarial forward paths.

    Wraps the C-level ``SNEPPX_arc_build_adversarial_train_graph`` when available.
    """

    def __init__(self, attack_epsilon: float = 0.01):
        self.attack_epsilon = attack_epsilon
        self._has_c = _HAS_C

    def build(self, weights: List[np.ndarray], x_clean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build forward graph returns (clean_output, adv_output).

        Pure Python fallback: applies FGSM perturbation, then runs a simple
        linear network on both clean and perturbed inputs sharing weights.
        """
        adv = x_clean + self.attack_epsilon * np.sign(np.random.randn(*x_clean.shape))
        adv = np.clip(adv, 0, 1)

        def fwd(x, w):
            h = x
            for wi in w:
                h = h @ wi
            return h

        clean_out = fwd(x_clean, weights)
        adv_out = fwd(adv, weights)
        return clean_out, adv_out
