"""Tests for algo_arc.py — ARC algorithm bindings."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

import numpy as np
from SneppX_ALG.interface_bindings import ARCAttackSim, ARCFForward, ARCGradientObfuscator, ARCInputGuard, ARCOutputVerifier


def test_attack_sim_pgd():
    attack = ARCAttackSim(eps=0.01, alpha=0.001, steps=3)
    x = np.random.randn(4, 10).astype(np.float32)
    y = np.random.randint(0, 2, 4)
    adv = attack.pgd(x, y, lambda x, y: np.sum(x))
    assert adv.shape == x.shape


def test_attack_sim_fgsm():
    attack = ARCAttackSim(eps=0.01)
    x = np.random.randn(2, 8).astype(np.float32)
    y = np.array([0, 1])
    adv = attack.fgsm(x, y, lambda x, y: np.sum(x))
    assert adv.shape == x.shape


def test_forward():
    fwd = ARCFForward()
    x = np.random.randn(3, 8).astype(np.float32)
    w = np.random.randn(4, 8).astype(np.float32)
    out = fwd.forward(x, w)
    assert out.shape == (3, 4)


def test_forward_with_bias():
    fwd = ARCFForward()
    x = np.random.randn(2, 5).astype(np.float32)
    w = np.random.randn(3, 5).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    out = fwd.forward(x, w, b)
    assert out.shape == (2, 3)


def test_forward_layers():
    fwd = ARCFForward()
    x = np.random.randn(2, 4).astype(np.float32)
    layers = [lambda z: z * 2, lambda z: z + 1]
    out = fwd.forward_layers(x, layers)
    assert out.shape == (2, 4)


def test_gradient_obfuscate_signed():
    obf = ARCGradientObfuscator()
    grad = np.random.randn(3, 3)
    obscured = obf.obfuscate(grad, "signed")
    assert np.all(np.abs(obscured) == 1)


def test_gradient_obfuscate_tanh():
    obf = ARCGradientObfuscator()
    grad = np.random.randn(3, 3)
    obscured = obf.obfuscate(grad, "tanh")
    assert obscured.shape == grad.shape


def test_input_guard_detect():
    guard = ARCInputGuard(threshold=2.0)
    x = np.random.randn(10, 5).astype(np.float32)
    stats = {"mean": np.zeros((1, 5)), "std": np.ones((1, 5))}
    flagged = guard.detect(x, stats)
    assert flagged.shape == (10,)


def test_input_guard_sanitize():
    guard = ARCInputGuard(threshold=1.0)
    x = np.array([[-10.0, 0.5, 10.0]])
    sanitized = guard.sanitize(x)
    assert np.all(sanitized >= -1.0)
    assert np.all(sanitized <= 1.0)


def test_output_verifier_certify():
    ov = ARCOutputVerifier()
    logits = np.array([[1.0, 0.5, 0.2], [0.3, 2.0, 0.1]], dtype=np.float32)
    certified, margins = ov.certify(logits, 0.5)
    assert certified.shape == (2,)
    assert margins.shape == (2,)


def test_output_verifier_worst_case():
    ov = ARCOutputVerifier()
    logits = np.array([[1.0, 0.5]], dtype=np.float32)
    wc = ov.worst_case(logits, 0.1)
    assert np.all(wc <= logits)


if __name__ == "__main__":
    test_attack_sim_pgd()
    test_attack_sim_fgsm()
    test_forward()
    test_forward_with_bias()
    test_forward_layers()
    test_gradient_obfuscate_signed()
    test_gradient_obfuscate_tanh()
    test_input_guard_detect()
    test_input_guard_sanitize()
    test_output_verifier_certify()
    test_output_verifier_worst_case()
    print("All algo_arc tests passed.")
