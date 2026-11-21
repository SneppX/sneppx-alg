"""Tests for Python algorithm wrappers (ARC/NPE/FM/Trainer).

NOTE: Run directly with `python test_algo_wrappers.py`, not via pytest.
"""
__test__ = False

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'bindings', 'python'))

import numpy as np

from SneppX_ALG.interface_bindings.algo_arc import (
    ARCAttackSim, ARCFForward, ARCGradientObfuscator, ARCInputGuard,
    ARCOutputVerifier, ARCAdversarialTrainGraph,
)
from SneppX_ALG.interface_bindings.algo_npe import NPECompiler, NPEInstruction, NPEProgram
from SneppX_ALG.interface_bindings.algo_fm import FMSync, FMSyncNCCL
from SneppX_ALG.interface_bindings.train import TrainConfig

tests_passed = 0
tests_failed = 0

def test(name, fn):
    global tests_passed, tests_failed
    try:
        fn()
        print(f"PASS: {name}")
        tests_passed += 1
    except Exception as e:
        print(f"FAIL: {name} — {e}")
        tests_failed += 1


def test_arc_adversarial_train_graph():
    builder = ARCAdversarialTrainGraph(attack_epsilon=0.01)
    weights = [np.random.randn(4, 8).astype(np.float32),
               np.random.randn(8, 2).astype(np.float32)]
    x = np.random.randn(3, 4).astype(np.float32)
    clean, adv = builder.build(weights, x)
    assert clean.shape == (3, 2), f"clean shape {clean.shape}"
    assert adv.shape == (3, 2), f"adv shape {adv.shape}"
    assert not np.allclose(clean, adv), "clean != adv"


def test_npe_jit_optimize():
    compiler = NPECompiler()
    prog = compiler.compile([
        {"type": "nop", "dest": "", "args": []},
        {"type": "matmul", "dest": "r1", "args": ["r0", "w1"]},
        {"type": "relu", "dest": "r2", "args": ["r1"]},
        {"type": "add", "dest": "r3", "args": ["r2", "b1"]},
    ])
    opt = compiler.jit_optimize(prog)
    assert len(opt) == 2, f"expected 2 instructions after optimize (matmul_relu, add), got {len(opt)}"
    assert opt[0].opcode == "matmul_relu", f"expected matmul_relu, got {opt[0].opcode}"
    assert opt[1].opcode == "add", f"expected add, got {opt[1].opcode}"


def test_fm_sync_nccl():
    sync = FMSyncNCCL()
    data = np.array([1.0, 2.0, 3.0])
    result = sync.sync(data)
    assert np.allclose(result, data), "sync preserves data"

    calls = []
    def cb(x):
        calls.append(x.copy())
    result2 = sync.sync(data, callback=cb)
    assert len(calls) == 1, "callback invoked"
    assert np.allclose(calls[0], data), "callback gets data"


def test_train_config_cuda_optimizer():
    cfg = TrainConfig()
    assert cfg.use_cuda_optimizer == 0, "default off"
    cfg.use_cuda_optimizer = True
    assert cfg.use_cuda_optimizer == 1, "set to 1"


def test_arc_input_guard():
    guard = ARCInputGuard(threshold=2.0)
    x = np.array([[0.0, 0.0], [10.0, -10.0]])
    stats = {"mean": np.array([[0.0, 0.0]]), "std": np.array([[1.0, 1.0]])}
    detected = guard.detect(x, stats)
    assert detected.shape == (2,), f"detect shape {detected.shape}"
    assert detected[1], "anomaly detected"
    sanitized = guard.sanitize(x)
    assert np.all(sanitized[0] == x[0]), "inlier unchanged"
    assert sanitized[1][0] == 2.0, "outlier clipped"


def test_arc_output_verifier():
    verifier = ARCOutputVerifier()
    logits = np.array([[1.0, 3.0, 2.0]])
    certified, margin = verifier.certify(logits, 0.5)
    assert certified[0], "certified"
    assert margin[0] > 0.5, "margin > radius"


def test_npe_compiler_basic():
    compiler = NPECompiler()
    prog = compiler.compile([
        {"type": "add", "dest": "r0", "args": ["a", "b"]},
        {"type": "mul", "dest": "r1", "args": ["r0", "c"]},
    ])
    assert len(prog) == 2
    assert prog[0].opcode == "add"
    assert prog[0].dest == "r0"


def test_fm_sync_all_reduce():
    sync = FMSync()
    t = np.array([1.0, 2.0, 3.0])
    r = sync.all_reduce(t, op="sum")
    assert np.allclose(r, t), "all_reduce identity"


if __name__ == "__main__":
    test("arc_adversarial_train_graph", test_arc_adversarial_train_graph)
    test("npe_jit_optimize", test_npe_jit_optimize)
    test("fm_sync_nccl", test_fm_sync_nccl)
    test("train_config_cuda_optimizer", test_train_config_cuda_optimizer)
    test("arc_input_guard", test_arc_input_guard)
    test("arc_output_verifier", test_arc_output_verifier)
    test("npe_compiler_basic", test_npe_compiler_basic)
    test("fm_sync_all_reduce", test_fm_sync_all_reduce)
    print(f"\n{tests_passed} passed, {tests_failed} failed")
    sys.exit(1 if tests_failed > 0 else 0)
