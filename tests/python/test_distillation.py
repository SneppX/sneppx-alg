"""Tests for Knowledge Distillation."""

import numpy as np
from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings.distillation import (
    kd_loss,
    attention_transfer_loss,
    feature_matching_loss,
    correlation_congruence_loss,
    hint_loss,
    crd_loss,
    multi_teacher_distillation_loss,
    ensemble_teacher_distillation,
    OnlineDistillation,
    DistillationPruner,
    distill_bert,
    distill_gpt,
)


def test_kd_loss_basic():
    student = Tensor.ones((2, 5))
    teacher = Tensor.ones((2, 5)) * 1.1
    loss = kd_loss(student, teacher, temperature=4.0, alpha=0.5)
    assert isinstance(loss, Tensor)
    assert loss.data.size == 1
    assert float(loss.data) >= 0.0


def test_kd_loss_with_labels():
    student = Tensor.ones((2, 5))
    teacher = Tensor.ones((2, 5)) * 1.1
    labels = Tensor.from_numpy(np.array([0, 1]))
    loss = kd_loss(student, teacher, temperature=4.0, alpha=0.5, labels=labels)
    assert float(loss.data) >= 0.0


def test_kd_loss_sum_reduction():
    student = Tensor.ones((4, 5))
    teacher = Tensor.ones((4, 5)) * 1.1
    loss = kd_loss(student, teacher, temperature=2.0, alpha=0.3, reduction="sum")
    assert float(loss.data) >= 0.0


def test_kd_loss_alpha_zero():
    student = Tensor.ones((2, 5))
    teacher = Tensor.ones((2, 5)) * 1.1
    loss = kd_loss(student, teacher, temperature=4.0, alpha=0.0)
    assert float(loss.data) >= 0.0


def test_attention_transfer_loss():
    s_attn = [Tensor.ones((1, 4, 4))]
    t_attn = [Tensor.ones((1, 4, 4)) * 1.2]
    loss = attention_transfer_loss(s_attn, t_attn, beta=1.0)
    assert isinstance(loss, Tensor)


def test_attention_transfer_loss_mismatch():
    s_attn = [Tensor.ones((1, 4, 4))]
    t_attn = [Tensor.ones((1, 4, 4)), Tensor.ones((1, 4, 4))]
    try:
        attention_transfer_loss(s_attn, t_attn)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_feature_matching_loss():
    s_feat = [Tensor.ones((2, 16))]
    t_feat = [Tensor.ones((2, 16)) * 1.1]
    loss = feature_matching_loss(s_feat, t_feat)
    assert isinstance(loss, Tensor)


def test_feature_matching_loss_weighted():
    s_feat = [Tensor.ones((2, 16)), Tensor.ones((2, 8))]
    t_feat = [Tensor.ones((2, 16)) * 1.1, Tensor.ones((2, 8)) * 1.05]
    loss = feature_matching_loss(s_feat, t_feat, weights=[0.7, 0.3])
    assert isinstance(loss, Tensor)


def test_correlation_congruence_loss():
    s_feat = Tensor.ones((2, 3, 4, 4))
    t_feat = Tensor.ones((2, 3, 4, 4)) * 1.1
    loss = correlation_congruence_loss(s_feat, t_feat)
    assert isinstance(loss, Tensor)


def test_hint_loss():
    s_feat = Tensor.ones((2, 16))
    t_feat = Tensor.ones((2, 16)) * 1.1
    loss = hint_loss(s_feat, t_feat, beta=1.0)
    assert float(loss.data) >= 0.0


def test_crd_loss():
    student_logits = Tensor.randn((4, 8))
    teacher_logits = Tensor.randn((4, 8))
    negative_logits = Tensor.randn((4, 4, 8))
    loss = crd_loss(student_logits, teacher_logits, negative_logits, temperature=0.1)
    assert float(loss.data) >= 0.0


def test_multi_teacher_distillation_loss():
    student = Tensor.ones((2, 5))
    teachers = [Tensor.ones((2, 5)) * 1.1, Tensor.ones((2, 5)) * 0.9]
    loss = multi_teacher_distillation_loss(student, teachers, temperature=4.0)
    assert float(loss.data) >= 0.0


def test_ensemble_teacher_distillation_mean():
    student = Tensor.ones((2, 5))
    teachers = [Tensor.ones((2, 5)) * 1.1, Tensor.ones((2, 5)) * 0.9]
    loss = ensemble_teacher_distillation(
        student, teachers, temperature=3.0, method="mean"
    )
    assert float(loss.data) >= 0.0


def test_ensemble_teacher_distillation_vote():
    student = Tensor.ones((2, 5))
    teachers = [Tensor.ones((2, 5)) * 1.1, Tensor.ones((2, 5)) * 0.9]
    loss = ensemble_teacher_distillation(
        student, teachers, temperature=3.0, method="vote"
    )
    assert float(loss.data) >= 0.0


def test_online_distillation():
    distill = OnlineDistillation(
        num_branches=3, temperature=4.0, distillation_weight=0.5
    )
    assert distill.temperature == 4.0
    assert distill.distillation_weight == 0.5
    branch_logits = [
        Tensor.ones((2, 5)),
        Tensor.ones((2, 5)) * 1.1,
        Tensor.ones((2, 5)) * 0.9,
    ]
    loss = distill.compute_loss(branch_logits)
    assert float(loss.data) >= 0.0


def test_distillation_pruner():
    pruner = DistillationPruner(
        target_sparsity=0.3, distill_temp=4.0, distill_alpha=0.5
    )
    assert pruner.target_sparsity == 0.3


def test_distill_bert_logit():
    def student(x):
        return x * 1.05

    def teacher(x):
        return x * 1.1

    x = Tensor.ones((4, 5))
    labels = Tensor.from_numpy(np.array([0, 1, 2, 3]))
    loader = [(x, labels), (x, labels)]
    report = distill_bert(
        None,
        None,
        student,
        teacher,
        loader,
        num_epochs=2,
        temperature=4.0,
        alpha=0.5,
        distillation_type="logit",
    )
    assert report["distillation_type"] == "logit"
    assert len(report["loss_history"]) == 2
    assert report["num_batches"] == 4
    assert all(np.isfinite(v) for v in report["loss_history"])
    assert report["student_model"] is student


def test_distill_bert_invalid_type():
    try:
        distill_bert(
            None, None, lambda x: x, lambda x: x, [], distillation_type="bogus"
        )
        assert False, "Should have raised"
    except ValueError:
        pass


def test_distill_bert_feature_matching():
    def student(x):
        return {"logits": x * 1.05, "hidden_states": [x * 0.9, x * 0.8]}

    def teacher(x):
        return {"logits": x * 1.1, "hidden_states": [x, x * 0.95]}

    x = Tensor.ones((4, 5))
    loader = [(x,)]
    report = distill_bert(
        None, None, student, teacher, loader, num_epochs=1, distillation_type="layer"
    )
    assert len(report["loss_history"]) == 1
    assert np.isfinite(report["loss_history"][0])
    # layer loss should exceed a pure logit loss (extra feature term is >= 0)
    logit_report = distill_bert(
        None, None, student, teacher, loader, num_epochs=1, distillation_type="logit"
    )
    assert report["loss_history"][0] >= logit_report["loss_history"][0]


def test_distill_gpt_basic():
    def student(x):
        return x * 1.05

    def teacher(x):
        return x * 1.1

    x = Tensor.ones((2, 6))
    labels = Tensor.from_numpy(np.array([0, 1]))
    report = distill_gpt(
        student, teacher, [(x, labels)], temperature=2.0, alpha=0.5, num_epochs=1
    )
    assert len(report["loss_history"]) == 1
    assert report["num_batches"] == 1
    assert np.isfinite(report["loss_history"][0])
    assert "perplexity" in report


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
    print(f"\n{'=' * 50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
