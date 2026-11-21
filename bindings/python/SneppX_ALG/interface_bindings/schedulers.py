"""Learning-rate schedulers — a comprehensive set used by modern LLM training.

Each scheduler exposes ``step()`` (advance one epoch or batch) and
``get_lr()``. All are framework-agnostic and operate purely on a base
``lr`` scalar, so they plug into any optimizer in the package.
"""

from typing import List, Optional, Callable
import math


class LRScheduler:
    """Base class for all schedulers."""

    def __init__(self, optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.base_lrs = list(
            getattr(optimizer, "param_groups", [{"lr": getattr(optimizer, "lr", 1e-3)}])
        )
        if not self.base_lrs:
            self.base_lrs = [1e-3]
        if self.base_lrs and isinstance(self.base_lrs[0], dict):
            self.base_lrs = [float(g.get("lr", 1e-3)) for g in self.base_lrs]
        else:
            self.base_lrs = [
                float(x) if not isinstance(x, (int, float)) else float(x)
                for x in self.base_lrs
            ]
        self.last_epoch = last_epoch
        self._initial_step()

    def _initial_step(self):
        self._step_count = 0
        self.step()

    def get_lr(self) -> float:
        return self.base_lrs[0]

    def step(self, epoch: Optional[int] = None):
        self._step_count += 1
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1
        lr = self.get_lr()
        # apply to optimizer (single-group simplified)
        if hasattr(self.optimizer, "lr"):
            self.optimizer.lr = lr
        if hasattr(self.optimizer, "set_lr"):
            self.optimizer.set_lr(lr)

    def state_dict(self) -> dict:
        return {"last_epoch": self.last_epoch, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict):
        self.last_epoch = state.get("last_epoch", -1)
        self.base_lrs = state.get("base_lrs", self.base_lrs)


class StepLR(LRScheduler):
    """Decay LR by ``gamma`` every ``step_size`` epochs."""

    def __init__(
        self, optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1
    ):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lrs[0] * (self.gamma ** (self.last_epoch // self.step_size))


class ExponentialLR(LRScheduler):
    """Multiply LR by ``gamma`` every epoch."""

    def __init__(self, optimizer, gamma: float = 0.95, last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lrs[0] * (self.gamma**self.last_epoch)


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing from ``eta_max`` to ``eta_min`` over ``T_max``."""

    def __init__(
        self, optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch >= self.T_max:
            return self.eta_min
        cos = math.cos(math.pi * (self.last_epoch % (2 * self.T_max)) / self.T_max)
        return self.eta_min + 0.5 * (self.base_lrs[0] - self.eta_min) * (1 + cos)


class CosineAnnealingWarmRestarts(LRScheduler):
    """Cosine annealing with warm restarts (SGDR)."""

    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        T_cur = self.last_epoch
        T_i = self.T_0
        while T_cur >= T_i and T_i > 0:
            T_cur -= T_i
            T_i = int(T_i * self.T_mult)
        if T_i <= 0:
            T_i = self.T_0
        cos = math.cos(math.pi * T_cur / T_i)
        return self.eta_min + 0.5 * (self.base_lrs[0] - self.eta_min) * (1 + cos)


class ConstantLRWithWarmup(LRScheduler):
    """Linear warmup to base lr, then constant."""

    def __init__(self, optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = max(1, warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch < self.warmup_steps:
            return self.base_lrs[0] * (self.last_epoch + 1) / self.warmup_steps
        return self.base_lrs[0]


class LinearWarmupCosineDecay(LRScheduler):
    """Linear warmup for ``warmup_steps`` then cosine decay to ``min_lr``.

    This is the de-facto standard LLM pretraining schedule.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(1, int(warmup_steps))
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        step = self.last_epoch
        base = self.base_lrs[0]
        if step < self.warmup_steps:
            return base * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        return self.min_lr + 0.5 * (base - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )


class PolynomialLR(LRScheduler):
    """Polynomial decay: ``lr = base * (1 - t/T)^power``."""

    def __init__(
        self,
        optimizer,
        total_steps: int,
        power: float = 2.0,
        lr_end: float = 0.0,
        last_epoch: int = -1,
    ):
        self.total_steps = max(1, total_steps)
        self.power = power
        self.lr_end = lr_end
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        t = min(self.last_epoch, self.total_steps)
        return (self.base_lrs[0] - self.lr_end) * (
            (1 - t / self.total_steps) ** self.power
        ) + self.lr_end


class OneCycleLR(LRScheduler):
    """1cycle policy: warmup to max_lr then cosine-anneal to min_lr."""

    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ):
        self.max_lr = max_lr
        self.total_steps = max(1, total_steps)
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        step = self.last_epoch
        if step == 0:
            return self.max_lr / self.div_factor
        if step >= self.total_steps:
            return self.max_lr / self.final_div_factor
        if step < self.total_steps * self.pct_start:
            # warmup phase
            pct = step / (self.total_steps * self.pct_start)
            return (self.max_lr / self.div_factor) + pct * (
                self.max_lr - self.max_lr / self.div_factor
            )
        else:
            # anneal phase
            pct = (step - self.total_steps * self.pct_start) / (
                self.total_steps * (1 - self.pct_start)
            )
            if self.anneal_strategy == "cos":
                return self.max_lr * (1 + math.cos(math.pi * pct)) / 2
            else:  # linear
                return self.max_lr * (1 - pct)


class ReduceLROnPlateau(LRScheduler):
    """Reduce LR when a monitored metric has stopped improving."""

    def __init__(
        self,
        optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8,
        last_epoch: int = -1,
    ):
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.best = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        super().__init__(optimizer, last_epoch)

    def is_better(self, current: float) -> bool:
        if self.mode == "min":
            if self.threshold_mode == "rel":
                return current < self.best * (1 - self.threshold)
            else:
                return current < self.best - self.threshold
        else:
            if self.threshold_mode == "rel":
                return current > self.best * (1 + self.threshold)
            else:
                return current > self.best + self.threshold

    def step(self, metrics: Optional[float] = None, epoch: Optional[int] = None):
        if metrics is None:
            # behave like a fixed scheduler if no metric provided
            super().step(epoch)
            return
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs += 1
        if self.is_better(metrics):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs > self.patience:
            new_lr = max(self.min_lr, self.base_lrs[0] * self.factor)
            self.base_lrs = [new_lr]
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        super().step(epoch)


class CosineAnnealingWithWarmupRestarts(LRScheduler):
    """Wrapper combining ConstantLRWithWarmup + CosineAnnealingWarmRestarts."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup = ConstantLRWithWarmup(optimizer, warmup_steps, last_epoch - 1)
        self.cosine = CosineAnnealingWarmRestarts(
            optimizer, T_0, T_mult, eta_min, last_epoch
        )
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch < self.warmup.warmup_steps:
            return self.warmup.get_lr()
        return self.cosine.get_lr()


class SequentialLR(LRScheduler):
    """Chain multiple schedulers, switching at ``milestones``."""

    def __init__(
        self,
        optimizer,
        schedulers: List[LRScheduler],
        milestones: List[int],
        last_epoch: int = -1,
    ):
        self.schedulers = schedulers
        self.milestones = milestones
        super().__init__(optimizer, last_epoch)

    def _active(self) -> LRScheduler:
        idx = 0
        for m in self.milestones:
            if self.last_epoch >= m:
                idx += 1
        idx = min(idx, len(self.schedulers) - 1)
        return self.schedulers[idx]

    def get_lr(self) -> float:
        return self._active().get_lr()

    def step(self, epoch: Optional[int] = None):
        self._step_count += 1
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1
        self._active().step(epoch)
        lr = self.get_lr()
        if hasattr(self.optimizer, "lr"):
            self.optimizer.lr = lr


class ChainedScheduler(LRScheduler):
    """Apply a list of schedulers sequentially on each step."""

    def __init__(self, optimizer, schedulers: List[LRScheduler], last_epoch: int = -1):
        self.schedulers = schedulers
        super().__init__(optimizer, last_epoch)

    def step(self, epoch: Optional[int] = None):
        for s in self.schedulers:
            s.step(epoch)
        lr = self.schedulers[-1].get_lr()
        if hasattr(self.optimizer, "lr"):
            self.optimizer.lr = lr

    def get_lr(self) -> float:
        return self.schedulers[-1].get_lr()


class TriStageLR(LRScheduler):
    """Three-stage LR: warmup -> constant -> linear decay (T5-style)."""

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_fraction: float = 0.1,
        decay_fraction: float = 0.1,
        last_epoch: int = -1,
    ):
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(1, int(total_steps * warmup_fraction))
        self.decay_start = total_steps - max(1, int(total_steps * decay_fraction))
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        step = self.last_epoch
        base = self.base_lrs[0]
        if step < self.warmup_steps:
            return base * (step + 1) / self.warmup_steps
        if step > self.decay_start:
            progress = (step - self.decay_start) / max(
                1, self.total_steps - self.decay_start
            )
            return base * (1 - progress)
        return base


class CosineAnnealingWithWarmup(LRScheduler):
    """Cosine annealing with linear warmup (standard LLM training schedule)."""

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(1, warmup_steps)
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        step = self.last_epoch
        base = self.base_lrs[0]
        if step < self.warmup_steps:
            return base * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        return self.eta_min + 0.5 * (base - self.eta_min) * (
            1 + math.cos(math.pi * progress)
        )


def get_scheduler(name: str, optimizer, **kwargs) -> LRScheduler:
    """Factory returning a scheduler by name."""
    registry = {
        "step": StepLR,
        "exponential": ExponentialLR,
        "cosine": CosineAnnealingLR,
        "cosine_warm_restarts": CosineAnnealingWarmRestarts,
        "warmup_constant": ConstantLRWithWarmup,
        "linear_warmup_cosine": LinearWarmupCosineDecay,
        "polynomial": PolynomialLR,
        "onecycle": OneCycleLR,
        "reduce_on_plateau": ReduceLROnPlateau,
        "tri_stage": TriStageLR,
    }
    if name not in registry:
        raise ValueError(f"Unknown scheduler '{name}'. Available: {list(registry)}")
    return registry[name](optimizer, **kwargs)
