"""Trainer v3 — autograd-powered training loop with YAML config and Phase 4 infra."""

import os
import math
import time
import json
import random
from typing import Optional, Callable, Dict, Any, List, Union, Tuple

import numpy as np

from .tensor import Tensor
from .nn import Module
from .optim import Optimizer, SGD, AdamW, Lion, LAMB
from .schedulers import (
    LRScheduler,
    get_scheduler,
    CosineAnnealingLR,
    LinearWarmupCosineDecay,
)
from .amp import GradScaler, autocast

# ===========================================================================
#  Config
# ===========================================================================


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class TrainConfig:
    """Training configuration loaded from a dict or YAML file.

    Accessible as attributes / dict items for convenience.
    """

    DEFAULTS = {
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 100000,
            "max_epochs": 0,
            "seed": 42,
            "device": "cpu",
            "max_grad_norm": 1.0,
            "label_smoothing": 0.0,
            "log_every": 10,
            "eval_every": 500,
            "save_every": 5000,
            "checkpoint_dir": "checkpoints",
            "resume_from": None,
        },
        "optimizer": {
            "name": "adamw",
            "lr": 3e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "momentum": 0.0,
        },
        "scheduler": {
            "name": "cosine",
            "warmup_steps": 0,
            "min_lr": 0.0,
            "t_max": None,
        },
        "amp": {
            "enabled": False,
            "dtype": "float16",
            "init_scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
        },
        "logging": {
            "use_wandb": False,
            "wandb_project": None,
            "wandb_run_name": None,
        },
        "checkpoint": {
            "keep_last_n": 5,
        },
        "distributed": {
            "enabled": False,
            "device": "cpu",
            "sync_bn": False,
        },
        "profiler": {
            "enabled": False,
            "flop_estimation": False,
            "save_path": None,
        },
        "privacy": {
            "enabled": False,
            "noise_multiplier": 1.0,
            "max_per_sample_grad_norm": 1.0,
            "epsilon": 8.0,
            "delta": 1e-5,
            "accountant": "rdp",
            "num_samples": None,
        },
    }

    def __init__(self, config_dict: Optional[dict] = None):
        self._data = _deep_merge(self.DEFAULTS, config_dict or {})

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(data)

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(data)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._data:
            val = self._data[name]
            if isinstance(val, dict):
                return _ConfigView(val)
            return val
        return _ConfigView({})

    def to_dict(self) -> dict:
        return self._data.copy()

    def override(self, overrides: Dict[str, Any]):
        self._data = _deep_merge(self._data, overrides)
        return self


class _ConfigView:
    """Read-only view into a nested dict with attribute access."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        val = self._data.get(name)
        if isinstance(val, dict):
            return _ConfigView(val)
        return val

    def __repr__(self):
        return repr(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def get(self, key, default=None):
        val = self._data.get(key)
        if val is None:
            return default
        if isinstance(val, dict):
            return _ConfigView(val)
        return val


# ===========================================================================
#  Gradient utilities
# ===========================================================================


def clip_grad_norm_(
    params: List[Tensor], max_norm: float, norm_type: float = 2.0
) -> float:
    """Clip gradient norms in-place. Returns total norm before clipping."""
    if max_norm <= 0:
        return 0.0
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    total_norm = 0.0
    for p in params:
        g = p.grad.data
        total_norm += np.sum(g.astype(np.float64) ** norm_type)
    total_norm = float(total_norm ** (1.0 / norm_type))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad.data = p.grad.data * clip_coef
    return total_norm


def estimate_flops(model: Module, input_shape: Tuple[int, ...]) -> int:
    """Rough FLOP estimate for a Module based on parameter counts."""
    total_params = sum(p.numel for p in model.parameters())
    if len(input_shape) >= 2:
        batch = input_shape[0]
        seq = input_shape[1] if len(input_shape) > 1 else 1
    else:
        batch = 1
        seq = input_shape[0] if input_shape else 1
    embed = int(math.sqrt(total_params / (seq * seq))) if total_params > 0 else 1
    forward_flops = total_params * seq * 2
    backward_flops = forward_flops * 2
    return int(forward_flops + backward_flops)


# ===========================================================================
#  Trainer
# ===========================================================================


class Trainer:
    """Autograd-powered training loop with checkpoint, DDP, and profiler support.

    Usage:
        model = MyModule()
        trainer = Trainer(model, config)
        trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: Module,
        config: Union[TrainConfig, dict, str],
        loss_fn: Optional[Callable] = None,
        distributed_wrapper=None,
        checkpoint_manager=None,
        profiler=None,
        experiment_tracker=None,
    ):
        if isinstance(config, str):
            if config.endswith(".yaml") or config.endswith(".yml"):
                self.config = TrainConfig.from_yaml(config)
            elif config.endswith(".json"):
                self.config = TrainConfig.from_json(config)
            else:
                self.config = TrainConfig.from_yaml(config)
        elif isinstance(config, dict):
            self.config = TrainConfig(config)
        else:
            self.config = config

        self.loss_fn = loss_fn or self._default_loss
        self._step = 0
        self._epoch = 0
        self._best_metric = None
        self._stopped = False
        self._t0 = time.time()

        self._setup_seed()

        # DDP wrapper
        dist_cfg = self.config.distributed
        if distributed_wrapper is not None:
            self.model = distributed_wrapper
        elif dist_cfg and dist_cfg.enabled:
            from .distributed_wrapper import DistributedWrapper

            self.model = DistributedWrapper(
                model,
                device=dist_cfg.device or "cpu",
                sync_bn=dist_cfg.sync_bn or False,
            )
        else:
            self.model = model

        # Move model to device
        device = self.config.training.device or "cpu"
        if device != "cpu" and hasattr(self.model, "to"):
            self.model.to(device)

        self._setup_params()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = self._build_scaler()

        # Privacy
        priv_cfg = self.config.privacy
        if priv_cfg and priv_cfg.enabled:
            from .differential_privacy import DPSGD, RDPAccountant

            self._dpsgd = DPSGD(
                self.optimizer,
                noise_multiplier=float(priv_cfg.noise_multiplier),
                max_grad_norm=float(priv_cfg.max_per_sample_grad_norm),
                num_samples=int(priv_cfg.num_samples) if priv_cfg.num_samples else None,
                epsilon=float(priv_cfg.epsilon),
                delta=float(priv_cfg.delta),
            )
            self.optimizer = self._dpsgd
        else:
            self._dpsgd = None

        # Checkpoint manager
        ckpt_cfg = self.config.checkpoint
        if checkpoint_manager is not None:
            self.checkpoint_manager = checkpoint_manager
        else:
            from .checkpoint_manager import CheckpointManager

            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=self.config.training.checkpoint_dir or "checkpoints",
                keep_last=int(ckpt_cfg.keep_last_n or 5),
                keep_best=int(getattr(ckpt_cfg, "keep_best", 3) or 3),
                async_save=True,
            )

        # Profiler
        prof_cfg = self.config.profiler
        if profiler is not None:
            self.profiler = profiler
        elif prof_cfg and prof_cfg.enabled:
            from .profiler import TrainProfiler

            self.profiler = TrainProfiler()
        else:
            self.profiler = None

        # Experiment tracker
        log_cfg = self.config.logging
        if experiment_tracker is not None:
            self.tracker = experiment_tracker
        elif log_cfg and log_cfg.use_wandb:
            from .experiment_tracker import (
                WandbTracker,
                ExperimentTracker,
                CompositeTracker,
            )

            comp = CompositeTracker()
            comp.add(
                ExperimentTracker(
                    storage_dir=self.config.training.checkpoint_dir or "experiments"
                )
            )
            comp.add(
                WandbTracker(
                    project=log_cfg.wandb_project or "sneppx",
                    name=log_cfg.wandb_run_name,
                )
            )
            self.tracker = comp
        else:
            from .experiment_tracker import ExperimentTracker

            self.tracker = ExperimentTracker(
                storage_dir=self.config.training.checkpoint_dir or "experiments"
            )

    def _default_loss(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return outputs.mse_loss(targets)

    def _setup_seed(self):
        seed = self.config.training.seed
        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def _setup_params(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def _build_optimizer(self) -> Optimizer:
        cfg = self.config.optimizer
        params = list(self.model.parameters())
        name = cfg.name.lower()
        lr = float(cfg.lr)
        wd = float(cfg.weight_decay)
        eps = float(cfg.eps)
        betas = tuple(float(b) for b in cfg.betas)
        momentum = float(getattr(cfg, "momentum", 0.0))

        if name == "adamw":
            return AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
        elif name == "sgd":
            return SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
        elif name == "lion":
            return Lion(params, lr=lr, betas=betas, weight_decay=wd)
        elif name == "lamb":
            return LAMB(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {name}")

    def _build_scheduler(self):
        cfg = self.config.scheduler
        name = cfg.name.lower()
        if name == "none" or name is None:
            return None

        sch_cfg = {
            "optimizer": self.optimizer,
            "warmup_steps": int(cfg.warmup_steps or 0),
            "total_steps": int(cfg.t_max or self.config.training.max_steps),
            "min_lr": float(cfg.min_lr or 0.0),
        }
        if name == "cosine":
            return LinearWarmupCosineDecay(**sch_cfg)
        elif name == "constant":
            from .schedulers import ConstantLRWithWarmup

            return ConstantLRWithWarmup(**sch_cfg)
        else:
            return get_scheduler(name, self.optimizer)

    def _build_scaler(self):
        cfg = self.config.amp
        if not cfg.enabled:
            return None
        return GradScaler(
            init_scale=cfg.init_scale,
            growth_factor=cfg.growth_factor,
            backoff_factor=cfg.backoff_factor,
            growth_interval=cfg.growth_interval,
            enabled=True,
        )

    @property
    def step(self) -> int:
        return self._step

    @property
    def epoch(self) -> int:
        return self._epoch

    # ---- Training step ----

    def train_step(self, batch) -> Tensor:
        """Run one training step (forward + backward + update)."""
        step_t0 = time.perf_counter()
        cfg = self.config.training
        self.model.train()

        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            inputs = batch
            targets = None

        # ---- DP-SGD path (per-sample losses) ----
        if self._dpsgd is not None:
            outputs = self.model(inputs)
            batch_size = outputs.shape[0] if hasattr(outputs, "shape") else len(outputs)
            per_sample = []
            for i in range(batch_size):
                if targets is not None:
                    if hasattr(targets, "shape") and len(targets.shape) > 0 and targets.shape[0] == batch_size:
                        t = targets[i : i + 1]
                    else:
                        t = targets
                else:
                    t = None
                if t is not None:
                    samp = outputs[i : i + 1].mse_loss(t)
                else:
                    samp = outputs[i : i + 1].sum()
                per_sample.append(samp)
            self._dpsgd.step(per_sample)
            loss = sum(per_sample) / batch_size

            if self.scheduler is not None:
                self.scheduler.step()
            self._step += 1
            step_elapsed = time.perf_counter() - step_t0
            if self.profiler is not None:
                self.profiler.record_step(step_elapsed)
                if hasattr(self.profiler, "memory") and hasattr(
                    self.profiler.memory, "update"
                ):
                    total_params = sum(p.numel for p in self.model.parameters())
                    self.profiler.memory.update(total_params * 4)
            return loss

        # ---- Standard path ----
        if self.scaler is not None:
            with autocast(enabled=True, dtype=self.config.amp.dtype):
                outputs = self.model(inputs)
                loss = (
                    self.loss_fn(outputs, targets) if targets is not None else outputs
                )
                loss = self.scaler.scale_tensor(loss)
        else:
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets) if targets is not None else outputs

        loss.backward()

        if self.scaler is not None:
            self.scaler.unscale_([p.grad for p in self.model.parameters()])

        # DDP gradient sync
        if hasattr(self.model, "sync_gradients"):
            self.model.sync_gradients()

        if cfg.max_grad_norm > 0:
            clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)

        if (self._step + 1) % cfg.gradient_accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

        self._step += 1

        step_elapsed = time.perf_counter() - step_t0
        if self.profiler is not None:
            self.profiler.record_step(step_elapsed)
            if hasattr(self.profiler, "memory") and hasattr(
                self.profiler.memory, "update"
            ):
                total_params = sum(p.numel for p in self.model.parameters())
                self.profiler.memory.update(total_params * 4)

        return loss

    # ---- Evaluation step ----

    def eval_step(self, batch) -> Tensor:
        """Run one evaluation step (no gradient tracking)."""
        self.model.eval()
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            inputs = batch
            targets = None
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets) if targets is not None else outputs
        return loss

    # ---- Fit ----

    def fit(
        self,
        train_loader,
        val_loader=None,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        """Main training loop.

        Args:
            train_loader: iterable yielding batches.
            val_loader: optional iterable for evaluation.
            max_steps: override config max_steps.
            max_epochs: override config max_epochs.
            callbacks: list of ``fn(trainer)`` called after each step.
        """
        cfg = self.config.training
        max_steps = max_steps or cfg.max_steps
        max_epochs = max_epochs or cfg.max_epochs or 0
        callbacks = callbacks or []
        train_iter = iter(train_loader)

        if cfg.resume_from:
            self.checkpoint_manager.load(self, cfg.resume_from)
        else:
            self.checkpoint_manager.load_latest(self)

        self._step = 0
        self._epoch = 0
        self._stopped = False
        self._t0 = time.time()

        # Start experiment tracking
        run_name = getattr(self.config.training, "run_name", "")
        params = {
            "optimizer": self.config.optimizer.name,
            "lr": self.config.optimizer.lr,
            "batch_size": cfg.batch_size,
            "max_steps": max_steps,
            "model_params": sum(p.numel for p in self.model.parameters()),
        }
        self.tracker.start_run(
            experiment_name=getattr(self.config.training, "experiment_name", "sneppx"),
            run_name=run_name or f"run_{int(time.time())}",
            params=params,
            config=self.config.to_dict(),
        )

        # FLOP estimation
        if self.profiler is not None and hasattr(train_loader, "__iter__"):
            try:
                sample = next(iter(train_loader))
                if isinstance(sample, (list, tuple)):
                    inp = sample[0]
                else:
                    inp = sample
                if hasattr(inp, "shape"):
                    flops = estimate_flops(self.model, inp.shape)
                    self.profiler.profiler.record("flops_estimate", flops)
                    print(f"  Estimated FLOPs/step: {flops:,}")
            except Exception:
                pass

        while self._step < max_steps and not self._stopped:
            self._epoch += 1

            if max_epochs > 0 and self._epoch > max_epochs:
                break

            epoch_loss = 0.0
            epoch_batches = 0

            while self._step < max_steps:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                loss = self.train_step(batch)
                epoch_loss += float(loss.data.flat[0])
                epoch_batches += 1

                if self._step % cfg.log_every == 0:
                    avg_loss = epoch_loss / max(epoch_batches, 1)
                    elapsed = time.time() - self._t0
                    lr = self.optimizer.lr
                    step_s = self._step
                    s_per_sec = step_s / max(elapsed, 1e-8)
                    self.tracker.log_metrics(
                        {
                            "loss": avg_loss,
                            "lr": lr,
                            "steps_per_sec": s_per_sec,
                        },
                        step=step_s,
                    )
                    print(
                        f"Step {step_s}/{max_steps} | "
                        f"Epoch {self._epoch} | "
                        f"Loss {avg_loss:.4f} | "
                        f"LR {lr:.2e} | "
                        f"{s_per_sec:.1f} steps/s | "
                        f"Time {elapsed:.1f}s"
                    )

                for cb in callbacks:
                    cb(self)

                val_loss = None
                if (
                    cfg.eval_every > 0
                    and self._step % cfg.eval_every == 0
                    and val_loader is not None
                ):
                    val_loss = self.evaluate(val_loader)
                    self.tracker.log_metrics({"val_loss": val_loss}, step=self._step)
                    print(f"  Eval loss: {val_loss:.4f}")

                if cfg.save_every > 0 and self._step % cfg.save_every == 0:
                    metric = (
                        val_loss
                        if val_loss is not None
                        else (epoch_loss / max(epoch_batches, 1))
                    )
                    is_best = self._best_metric is None or (
                        val_loss is not None and val_loss < self._best_metric
                    )
                    if is_best and val_loss is not None:
                        self._best_metric = val_loss
                    self.checkpoint_manager.save(self, metric=metric, is_best=is_best)

        self.tracker.set_status("completed")
        self.tracker.end_run()

    # ---- Evaluate ----

    def evaluate(self, val_loader) -> float:
        """Run evaluation over ``val_loader``, return average loss."""
        total_loss = 0.0
        count = 0
        for batch in val_loader:
            loss = self.eval_step(batch)
            total_loss += float(loss.data.flat[0])
            count += 1
        return total_loss / max(count, 1)

    # ---- Checkpoint ----

    def save(self, path: str):
        """Direct pickle save (legacy). Prefer checkpoint_manager.save()."""
        self.checkpoint_manager.save(self)
        print(f"Checkpoint saved via manager: {path}")

    def load(self, path: str):
        """Direct pickle load (legacy). Prefer checkpoint_manager.load()."""
        self.checkpoint_manager.load(self, path)

    # ---- Profiler ----

    def print_profile(self):
        if self.profiler is not None:
            self.profiler.summary()
            prof_cfg = self.config.profiler
            if prof_cfg and prof_cfg.save_path:
                self.profiler.profiler.save_json(prof_cfg.save_path)


__all__ = ["TrainConfig", "Trainer", "clip_grad_norm_", "estimate_flops"]
