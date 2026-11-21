"""Tests for Model Pruning."""

import numpy as np
from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings.pruning import (
    magnitude_prune, l1_channel_prune, taylor_pruning,
    global_magnitude_prune, movement_pruning, soft_pruning,
    compute_sparsity, count_parameters, print_pruning_summary,
    apply_pruning_mask, recover_pruned_weights,
    distillation_loss, prune_and_distill, find_winning_ticket,
    rewrite_weights,
)
from SneppX_ALG.interface_bindings.nn import Linear, Sequential


def test_magnitude_prune_unstructured():
    w = Tensor.from_numpy(np.array([[1.0, 0.1, 0.5], [-0.2, 3.0, 0.05]]))
    pruned, mask = magnitude_prune(w, sparsity=0.5)
    assert pruned.shape == w.shape
    assert mask.shape == w.shape
    sparsity = compute_sparsity(pruned)
    assert abs(sparsity - 0.5) <= 0.2


def test_magnitude_prune_structured():
    w = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.1, 0.2, 0.3]]))
    pruned, mask = magnitude_prune(w, sparsity=0.3, structured=True, dim=0)
    assert pruned.shape == (3, 3)
    assert mask.shape == (3, 3)


def test_l1_channel_prune():
    w = Tensor.from_numpy(np.random.randn(4, 8).astype(np.float32))
    pruned, mask = l1_channel_prune(w, sparsity=0.25, dim=0)
    assert pruned.shape == (4, 8)
    assert compute_sparsity(pruned) >= 0.0


def test_taylor_pruning():
    w = Tensor.randn((4, 4))
    grad = Tensor.randn((4, 4)) * 0.1
    pruned, mask = taylor_pruning(w, grad, sparsity=0.5)
    assert pruned.shape == (4, 4)


def test_global_magnitude_prune():
    tensors = [Tensor.randn((4, 4)), Tensor.randn((8, 8))]
    pruned = global_magnitude_prune(tensors, sparsity=0.5)
    assert len(pruned) == 2


def test_movement_pruning():
    w = Tensor.from_numpy(np.random.randn(4, 4).astype(np.float32))
    grad = Tensor.from_numpy(np.ones((4, 4)).astype(np.float32))
    pruned, mask = movement_pruning(w, grad, sparsity=0.5)
    assert pruned.shape == (4, 4)


def test_soft_pruning():
    w = Tensor.randn((4, 4))
    pruned, mask = soft_pruning(w, sparsity=0.5, temperature=0.5)
    assert pruned.shape == (4, 4)


def test_compute_sparsity():
    w = Tensor.from_numpy(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32))
    sparsity = compute_sparsity(w)
    assert abs(sparsity - (1.0 - 2.0 / 6.0)) < 1e-6


def test_compute_sparsity_dense():
    w = Tensor.ones((3, 3))
    sparsity = compute_sparsity(w)
    assert sparsity == 0.0


def test_count_parameters():
    params = [Tensor.randn((4, 8)), Tensor.randn((8,)), Tensor.randn((8, 2)), Tensor.randn((2,))]
    total, nonzero = count_parameters(params)
    assert total == 4 * 8 + 8 + 8 * 2 + 2


def test_apply_pruning_mask():
    w = Tensor.randn((4, 4))
    mask_np = np.random.rand(4, 4) > 0.5
    pruned = apply_pruning_mask(w, mask_np)
    assert pruned.shape == (4, 4)


def test_recover_pruned_weights():
    original = Tensor.randn((4, 4))
    pruned, mask = magnitude_prune(original, sparsity=0.5)
    mask_np = mask.data.astype(bool)
    recovered = recover_pruned_weights(pruned, mask_np, original_weight=original)
    assert np.allclose(recovered.data, original.data)


def test_prune_and_distill():
    from SneppX_ALG.interface_bindings.pruning import PolynomialSparsityScheduler
    teacher_params = [Tensor.randn((4, 8)), Tensor.randn((8,)), Tensor.randn((8, 2)), Tensor.randn((2,))]
    student_params = [Tensor.randn((4, 8)), Tensor.randn((8,)), Tensor.randn((8, 2)), Tensor.randn((2,))]
    scheduler = PolynomialSparsityScheduler(total_steps=10, final_sparsity=0.5)
    pruned = prune_and_distill(student_params, teacher_params, dataloader=None, sparsity_scheduler=scheduler, epochs=2)
    assert len(pruned) == 4


def test_find_winning_ticket():
    params = [Tensor.randn((4, 4))]
    ticket, masks = find_winning_ticket(params, dataloader=None, sparsity=0.3, iterations=2)
    assert len(ticket) == 1
    assert len(masks) == 2


def test_rewrite_weights():
    w = Tensor.randn((4, 4))
    _, mask = magnitude_prune(w, sparsity=0.5)
    rewritten = rewrite_weights([w], [mask.data])
    assert rewritten[0].shape == (4, 4)


def test_distillation_loss():
    student_logits = Tensor.ones((2, 5))
    teacher_logits = Tensor.ones((2, 5)) * 1.1
    loss = distillation_loss(student_logits, teacher_logits, temperature=4.0)
    assert isinstance(loss, Tensor)
    assert loss.data.size == 1


def test_print_pruning_summary():
    params = [Tensor.randn((10, 10))]
    pruned, _ = magnitude_prune(params[0], sparsity=0.3)
    print_pruning_summary([pruned])


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
