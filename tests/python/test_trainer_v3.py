"""Tests for Trainer v3 (Autograd Training Loop)."""

from SneppX_ALG.interface_bindings.trainer_v3 import (
    TrainConfig, Trainer, clip_grad_norm_, estimate_flops,
)
from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings.nn import Linear, Sequential, Module
import numpy as np


class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(64, 10)

    def forward(self, x):
        return self.fc(x)


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg is not None


def test_train_config_from_dict():
    cfg = TrainConfig(config_dict={"learning_rate": 0.001, "batch_size": 8, "num_epochs": 2})
    assert cfg is not None


def test_train_config_attrs():
    cfg = TrainConfig(config_dict={"learning_rate": 0.01, "batch_size": 4})
    assert cfg.learning_rate == 0.01


def test_trainer_init():
    model = SimpleModel()
    cfg = TrainConfig(config_dict={"learning_rate": 0.001, "num_epochs": 1})
    trainer = Trainer(model, cfg)
    assert trainer is not None


def test_clip_grad_norm():
    params = [Tensor.randn((4, 4))]
    norm = clip_grad_norm_(params, max_norm=1.0)
    assert norm >= 0.0


def test_estimate_flops():
    model = SimpleModel()
    flops = estimate_flops(model, input_shape=(1, 64))
    assert flops > 0


if __name__ == "__main__":
    import sys
    locals_ = {k: v for k, v in locals().items() if k.startswith("test_")}
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        try:
            fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
