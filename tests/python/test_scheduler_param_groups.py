"""Regression: scheduler steps must update optimizer param_groups."""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python")
)

from SneppX_ALG.interface_bindings.schedulers import StepLR


class DummyOptimizer:
    def __init__(self):
        self.lr = 0.1
        self.param_groups = [{"lr": 0.1}, {"lr": 0.1}]

    def set_lr(self, lr):
        self.lr = lr


def test_step_updates_every_param_group():
    opt = DummyOptimizer()
    scheduler = StepLR(opt, step_size=1, gamma=0.5)

    # The base-class initialization performs the first step at epoch 0.
    assert [g["lr"] for g in opt.param_groups] == [0.1, 0.1]

    scheduler.step()
    assert [g["lr"] for g in opt.param_groups] == [0.05, 0.05]
    assert opt.lr == 0.05


def test_step_keeps_scalar_in_sync():
    opt = DummyOptimizer()
    scheduler = StepLR(opt, step_size=2, gamma=0.1)
    scheduler.step()
    scheduler.step()
    scheduler.step()
    assert opt.lr == pytest.approx(0.01)
    assert all(g["lr"] == pytest.approx(0.01) for g in opt.param_groups)
