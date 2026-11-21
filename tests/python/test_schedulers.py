"""Tests for LR Schedulers."""

import math
from SneppX_ALG.interface_bindings.schedulers import (
    LRScheduler, StepLR, ExponentialLR, CosineAnnealingLR,
    CosineAnnealingWarmRestarts, ConstantLRWithWarmup,
    LinearWarmupCosineDecay, PolynomialLR, OneCycleLR,
    ReduceLROnPlateau, SequentialLR, ChainedScheduler,
    TriStageLR, get_scheduler,
)


class DummyOptimizer:
    def __init__(self, lr=1e-3):
        self.lr = lr
        self.param_groups = [{"lr": lr}]

    def set_lr(self, lr):
        self.lr = lr


def test_lr_scheduler_base():
    opt = DummyOptimizer(1e-3)
    sched = LRScheduler(opt)
    assert sched.base_lrs == [1e-3]
    assert sched.get_lr() == 1e-3


def test_lr_scheduler_step():
    opt = DummyOptimizer(1e-3)
    sched = LRScheduler(opt)
    sched.step()
    assert opt.lr == 1e-3


def test_step_lr():
    opt = DummyOptimizer(1.0)
    sched = StepLR(opt, step_size=2, gamma=0.5)
    sched.step()
    assert opt.lr == 1.0
    sched.step()
    assert opt.lr == 0.5


def test_exponential_lr():
    opt = DummyOptimizer(1.0)
    sched = ExponentialLR(opt, gamma=0.9)
    sched.step()
    assert abs(opt.lr - 0.9) < 1e-10
    sched.step()
    assert abs(opt.lr - 0.81) < 1e-10


def test_cosine_annealing_lr():
    opt = DummyOptimizer(1.0)
    sched = CosineAnnealingLR(opt, T_max=4)
    sched.step()
    expected = 0.5 * (1.0 + math.cos(math.pi / 4))
    assert abs(opt.lr - expected) < 1e-6


def test_cosine_annealing_warm_restarts():
    opt = DummyOptimizer(1.0)
    sched = CosineAnnealingWarmRestarts(opt, T_0=2, T_mult=2)
    sched.step()
    sched.step()
    assert abs(opt.lr - 1.0) < 1e-5
    sched.step()
    sched.step()
    sched.step()
    sched.step()
    assert abs(opt.lr - 1.0) < 1e-5


def test_constant_lr_with_warmup():
    opt = DummyOptimizer(1.0)
    sched = ConstantLRWithWarmup(opt, warmup_steps=3)
    # After __init__ step, last_epoch=0 -> lr = 1.0 * (0+1)/3 = 0.333
    # First manual step: last_epoch=1 -> lr = 1.0 * (1+1)/3 = 0.666
    # Second: last_epoch=2 -> lr = 1.0 * (2+1)/3 = 1.0
    # Third: last_epoch=3 -> lr = 1.0 (constant)
    sched.step()
    assert abs(opt.lr - 2.0/3) < 1e-6
    sched.step()
    assert abs(opt.lr - 1.0) < 1e-6
    sched.step()
    assert abs(opt.lr - 1.0) < 1e-6


def test_linear_warmup_cosine_decay():
    opt = DummyOptimizer(1.0)
    sched = LinearWarmupCosineDecay(opt, total_steps=10, warmup_steps=3)
    for i in range(10):
        sched.step()
    assert opt.lr >= 0.0


def test_polynomial_lr():
    opt = DummyOptimizer(1.0)
    sched = PolynomialLR(opt, power=2.0, total_steps=4)
    sched.step()
    expected = 1.0 * (1 - 1/4) ** 2
    assert abs(opt.lr - expected) < 1e-6


def test_one_cycle_lr():
    opt = DummyOptimizer(0.0)
    sched = OneCycleLR(opt, max_lr=1.0, total_steps=6)
    for i in range(6):
        sched.step()
    assert opt.lr >= 0.0


def test_reduce_lr_on_plateau():
    opt = DummyOptimizer(1.0)
    sched = ReduceLROnPlateau(opt, factor=0.5, patience=2)
    # init step sets last_epoch=0
    # Three bad steps (non-improving) should trigger reduction
    sched.step(metrics=1.0)  # best=1.0, num_bad=0 (is better)
    sched.step(metrics=1.1)  # num_bad=1
    sched.step(metrics=1.2)  # num_bad=2
    sched.step(metrics=1.3)  # num_bad=3 > patience=2 -> reduce
    assert abs(opt.lr - 0.5) < 1e-6


def test_sequential_lr():
    opt = DummyOptimizer(1.0)
    sched1 = ExponentialLR(opt, gamma=0.5)
    sched2 = ExponentialLR(opt, gamma=0.1)
    sched = SequentialLR(opt, schedulers=[sched1, sched2], milestones=[2])
    # init step: last_epoch=0, active=s1, s1.step -> s1.last_epoch=1, lr=0.5
    # manual step 1: last_epoch=1, active=s1, s1.step -> s1.last_epoch=2, lr=0.25
    # manual step 2: last_epoch=2, active=s2 (2>=2), s2.step -> lr=0.1
    sched.step()
    assert abs(opt.lr - 0.25) < 1e-6
    sched.step()
    assert abs(opt.lr - 0.1) < 1e-6


def test_chained_scheduler():
    opt = DummyOptimizer(1.0)
    sched1 = StepLR(opt, step_size=1, gamma=0.5)
    sched2 = ExponentialLR(opt, gamma=0.5)
    chained = ChainedScheduler(opt, schedulers=[sched1, sched2])
    chained.step()
    assert opt.lr == 0.25


def test_tri_stage_lr():
    opt = DummyOptimizer(1.0)
    sched = TriStageLR(opt, total_steps=10, warmup_fraction=0.2, decay_fraction=0.2)
    for i in range(10):
        sched.step()
    assert opt.lr >= 0.0


def test_scheduler_state_dict():
    opt = DummyOptimizer(1.0)
    sched = StepLR(opt, step_size=2, gamma=0.5)
    sched.step()
    sd = sched.state_dict()
    assert "last_epoch" in sd
    assert "base_lrs" in sd


def test_get_scheduler():
    opt = DummyOptimizer(0.0)
    sched = get_scheduler("cosine", opt, T_max=10)
    assert isinstance(sched, CosineAnnealingLR)


def test_get_scheduler_linear_warmup():
    opt = DummyOptimizer(1.0)
    sched = get_scheduler("linear_warmup_cosine", opt, warmup_steps=2, total_steps=10)
    assert isinstance(sched, LinearWarmupCosineDecay)
    assert opt.lr > 0.0


if __name__ == "__main__":
    import sys
    locals_ = locals().copy()
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        if name.startswith("test_"):
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
