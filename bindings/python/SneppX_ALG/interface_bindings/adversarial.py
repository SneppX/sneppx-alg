"""Adversarial Robustness — FGSM/PGD attacks and adversarial training.

Provides gradient-based white-box attacks (Fast Gradient Sign Method and
Projected Gradient Descent) plus an ``AdversarialTrainer`` that augments a
normal training loop with adversarial examples to improve model robustness.
"""

import numpy as np
from typing import Optional, Callable, Any, Union, List

from .tensor import Tensor
from .nn import Module
from .optim import Optimizer


# ===========================================================================
#  Attacks
# ===========================================================================


class Attack:
    """Base class for white-box gradient-based attacks.

    Subclasses implement :meth:`generate`, returning adversarial inputs that
    are detached (no attack-time autograd graph attached).
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        targeted: bool = False,
    ):
        if epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

    def _input_grad(
        self,
        model: Module,
        inputs: Tensor,
        labels: Any,
        loss_fn: Callable[[Any, Any], Tensor],
    ) -> np.ndarray:
        """Run a single forward/backward to obtain the input gradient."""
        if not inputs.requires_grad:
            inputs.requires_grad_(True)
        # Clear any stale model grads from a previous call.
        for p in model.parameters():
            if p.grad is not None:
                p.zero_grad_()
        inputs.zero_grad_()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        if not isinstance(loss, Tensor):
            loss = Tensor(np.asarray(loss, dtype=np.float32))
        loss.backward()
        grad = inputs.grad
        if grad is None:
            raise RuntimeError("Input gradient is None; ensure inputs require grad.")
        return grad.data

    def _project(
        self, adv: np.ndarray, base: np.ndarray, epsilon: Optional[float] = None
    ) -> np.ndarray:
        eps = self.epsilon if epsilon is None else epsilon
        if eps is not None:
            adv = np.clip(adv, base - eps, base + eps)
        if self.clip_min is not None and self.clip_max is not None:
            adv = np.clip(adv, self.clip_min, self.clip_max)
        return adv

    def generate(
        self,
        model: Module,
        inputs: Tensor,
        labels: Any,
        loss_fn: Callable[[Any, Any], Tensor],
    ) -> Tensor:
        raise NotImplementedError


class FGSM(Attack):
    """Fast Gradient Sign Method (Goodfellow et al., 2015).

    Single-step attack::

        adv = clip(inputs + epsilon * sign(grad_loss(inputs)))
    """

    def generate(
        self,
        model: Module,
        inputs: Tensor,
        labels: Any,
        loss_fn: Callable[[Any, Any], Tensor],
    ) -> Tensor:
        base = inputs.data
        grad = self._input_grad(model, inputs, labels, loss_fn)
        if self.targeted:
            sign = -np.sign(grad)
        else:
            sign = np.sign(grad)
        adv = base + self.epsilon * sign
        adv = self._project(adv, base)
        return Tensor(adv, requires_grad=False)


class PGD(Attack):
    """Projected Gradient Descent (Madry et al., 2018).

    Iterative variant of FGSM with an optional random restart and an
    L-infinity projection back onto the epsilon ball at every step.
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        alpha: float = 0.01,
        steps: int = 40,
        random_start: bool = True,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        targeted: bool = False,
    ):
        super().__init__(epsilon, clip_min, clip_max, targeted)
        if alpha <= 0:
            raise ValueError(f"alpha (step size) must be > 0, got {alpha}")
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}")
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def generate(
        self,
        model: Module,
        inputs: Tensor,
        labels: Any,
        loss_fn: Callable[[Any, Any], Tensor],
    ) -> Tensor:
        base = inputs.data.astype(np.float64)
        adv = base.copy()
        if self.random_start and self.epsilon > 0:
            noise = np.random.uniform(-self.epsilon, self.epsilon, size=base.shape)
            adv = self._project(adv + noise, base)

        for _ in range(self.steps):
            tmp = Tensor(adv, requires_grad=True)
            grad = self._input_grad(model, tmp, labels, loss_fn)
            if self.targeted:
                adv = adv - self.alpha * np.sign(grad)
            else:
                adv = adv + self.alpha * np.sign(grad)
            adv = self._project(adv, base)
        return Tensor(adv.astype(np.float32), requires_grad=False)


class AdversarialExample:
    """Container holding clean, adversarial inputs and attack metadata."""

    def __init__(self, clean: Tensor, adversarial: Tensor, epsilon: float):
        self.clean = clean
        self.adversarial = adversarial
        self.epsilon = epsilon

    def numpy(self):
        return self.clean.data, self.adversarial.data


# ===========================================================================
#  Adversarial Trainer
# ===========================================================================


class AdversarialTrainer:
    """Train a model with adversarial examples (adversarial training).

    Each training step optionally generates adversarial examples for the
    current batch (controlled by ``attack_ratio``) and computes the loss on
    those examples before calling ``optimizer.step()``.

    Args:
        model: The model ``Module`` to train.
        optimizer: An :class:`Optimizer` instance.
        attacker: An :class:`Attack` instance (e.g. ``PGD`` or ``FGSM``).
        loss_fn: ``loss_fn(outputs, labels) -> scalar Tensor``.
        attack_ratio: Probability that a given step uses adversarial
            examples; otherwise the clean batch is used.
        num_classes: Optional, used for robustness evaluation.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        attacker: Attack,
        loss_fn: Callable[[Any, Any], Tensor],
        attack_ratio: float = 1.0,
        num_classes: Optional[int] = None,
    ):
        if not (0.0 <= attack_ratio <= 1.0):
            raise ValueError("attack_ratio must be in [0, 1]")
        self.model = model
        self.optimizer = optimizer
        self.attacker = attacker
        self.loss_fn = loss_fn
        self.attack_ratio = attack_ratio
        self.num_classes = num_classes
        self._step = 0
        self.history: List[dict] = []

    # -- low level ----------------------------------------------------------

    def _zero_model_grads(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.zero_grad_()

    def _to_inputs(self, batch) -> tuple:
        """Split a batch into (inputs, labels).

        Supports either a ``(inputs, labels)`` tuple/list, or an object with
        ``.inputs`` / ``.labels`` attributes (e.g. a dataset sample).
        """
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        if hasattr(batch, "inputs") and hasattr(batch, "labels"):
            return batch.inputs, batch.labels
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def train_step(self, batch) -> dict:
        """Perform a single adversarial training step. Returns metrics."""
        inputs, labels = self._to_inputs(batch)
        self.optimizer.zero_grad()
        self._zero_model_grads()

        use_adv = (self.attack_ratio >= 1.0) or (
            self.attack_ratio > 0.0 and np.random.rand() < self.attack_ratio
        )

        if use_adv:
            adv_inputs = self.attacker.generate(self.model, inputs, labels, self.loss_fn)
            # Detach input grad state from the attack: train on adv examples.
            train_inputs = adv_inputs
            inputs_for_loss = adv_inputs
        else:
            train_inputs = inputs
            inputs_for_loss = inputs

        # Clear grads possibly set during attack (model params).
        self._zero_model_grads()

        outputs = self.model(inputs_for_loss)
        loss = self.loss_fn(outputs, labels)
        if not isinstance(loss, Tensor):
            loss = Tensor(np.asarray(loss, dtype=np.float32))
        loss.backward()
        self.optimizer.step()

        metrics = {
            "step": self._step,
            "loss": float(loss.data.item()) if hasattr(loss.data, "item") else float(loss.data),
            "adversarial": use_adv,
        }
        self._step += 1
        self.history.append(metrics)
        return metrics

    def fit(self, data, epochs: int = 1) -> List[dict]:
        """Train for ``epochs`` over an iterable ``data`` of batches."""
        all_metrics: List[dict] = []
        for _ in range(epochs):
            for batch in data:
                all_metrics.append(self.train_step(batch))
        return all_metrics

    # -- robustness evaluation ---------------------------------------------

    def evaluate_robustness(
        self, data, attacker: Optional[Attack] = None
    ) -> dict:
        """Measure clean and robust accuracy on ``data``.

        Returns a dict with ``clean_acc`` and ``robust_acc`` (fraction of
        examples where the model is correct on clean and adversarial inputs
        respectively). Requires ``num_classes`` to be set or inferable.
        """
        atk = attacker if attacker is not None else self.attacker
        correct_clean = 0
        correct_robust = 0
        total = 0

        was_training = getattr(self.model, "training", True)
        self.model.eval() if hasattr(self.model, "eval") else None

        try:
            for batch in data:
                inputs, labels = self._to_inputs(batch)
                n = _batch_size(inputs, labels)
                total += n

                clean_out = self.model(inputs)
                clean_pred = _predict(clean_out, labels)
                correct_clean += int(np.sum(clean_pred == _label_array(labels)))

                adv_inputs = atk.generate(self.model, inputs, labels, self.loss_fn)
                adv_out = self.model(adv_inputs)
                adv_pred = _predict(adv_out, labels)
                correct_robust += int(np.sum(adv_pred == _label_array(labels)))
        finally:
            if hasattr(self.model, "train"):
                self.model.train() if was_training else self.model.eval()

        return {
            "clean_acc": correct_clean / total if total else 0.0,
            "robust_acc": correct_robust / total if total else 0.0,
            "total": total,
        }

    # -- persistence --------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "step": self._step,
            "attack_ratio": self.attack_ratio,
            "epsilon": self.attacker.epsilon,
            "history": self.history[-100:],
        }


# ===========================================================================
#  Helpers
# ===========================================================================


def _batch_size(inputs, labels) -> int:
    if hasattr(labels, "shape"):
        return int(labels.shape[0]) if len(labels.shape) > 0 else 1
    try:
        return len(labels)
    except TypeError:
        return 1


def _label_array(labels):
    if isinstance(labels, Tensor):
        return np.asarray(labels.data).reshape(-1)
    return np.asarray(labels).reshape(-1)


def _predict(outputs, labels):
    """Return predicted class indices from model outputs / logits."""
    if isinstance(outputs, dict):
        logits = outputs.get("logits", None)
    else:
        logits = outputs
    if isinstance(logits, Tensor):
        arr = np.asarray(logits.data)
    else:
        arr = np.asarray(logits)
    if arr.ndim == 3:
        arr = arr[:, -1, :]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.argmax(arr, axis=-1).reshape(-1)
