import numpy as np

from SneppX_ALG import regularization as reg


def test_penalties():
    w = np.array([1.0, -2.0, 3.0])
    assert reg.l1_penalty(w) == 6.0
    assert reg.l2_penalty(w) == 14.0
    assert reg.elastic_net(w, 0.5, 0.5) == 0.5 * 6.0 + 0.5 * 14.0
    assert abs(reg.group_lasso([np.array([3.0, 4.0])]) - 5.0) < 1e-9
    assert reg.total_variation(np.array([0.0, 1.0, 3.0])) == 3.0
    assert reg.proximity_penalty(w, w) == 0.0


def test_orthogonality_penalty():
    W = np.eye(3)
    assert reg.orthogonality_penalty(W) == 0.0
    assert reg.orthogonality_penalty(2 * np.eye(2)) > 0.0


def test_spectral_norm():
    A = np.diag([3.0, 1.0, 0.5])
    assert abs(reg.spectral_norm(A, num_iters=50) - 3.0) < 1e-6
    normed = reg.spectral_normalize(A, num_iters=50)
    assert abs(reg.spectral_norm(normed, num_iters=50) - 1.0) < 1e-6


def test_label_smoothing():
    oh = np.array([[1.0, 0.0], [0.0, 1.0]])
    sm = reg.smooth_one_hot(oh, eps=0.1)
    assert np.allclose(sm, [[0.95, 0.05], [0.05, 0.95]])
    assert np.allclose(sm.sum(axis=1), 1.0)
    sm2 = reg.smooth_labels([0, 1], num_classes=2, eps=0.2)
    assert np.allclose(sm2.sum(axis=1), 1.0)


def test_early_stopping():
    es = reg.EarlyStopping(patience=2, mode="min")
    assert es.step(1.0) is False
    assert es.step(0.5) is False
    assert es.step(0.6) is False
    assert es.step(0.7) is True
    assert es.best == 0.5


def test_weight_averaging():
    wa = reg.WeightAveraging(decay=0.9)
    assert np.allclose(wa.update([1.0, 1.0]), [1.0, 1.0])
    assert np.allclose(wa.update([3.0, 3.0]), [1.2, 1.2])


def test_mixup():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    xm, ym, lam = reg.mixup(x, y, alpha=0.2, seed=0)
    assert xm.shape == x.shape and ym.shape == y.shape
    assert 0.0 < lam < 1.0
    x1, y1, lam1 = reg.mixup(x, y, alpha=0.0, seed=0)
    assert np.allclose(x1, x) and lam1 == 1.0


def test_cutmix():
    x = np.random.default_rng(0).random((4, 8, 8, 3))
    y = np.eye(3)[[0, 1, 2, 0]]
    xc, yc, lam = reg.cutmix(x, y, alpha=1.0, seed=0)
    assert xc.shape == x.shape and yc.shape == y.shape
    assert 0.0 <= lam <= 1.0


def test_weight_decay():
    p = np.array([1.0, 1.0])
    g = np.array([0.1, 0.1])
    out = reg.apply_weight_decay(p, g, lr=0.1, weight_decay=0.5)
    assert np.allclose(out, p - 0.1 * (g + 0.5 * p))
    out2 = reg.apply_decoupled_weight_decay(p, g, lr=0.1, weight_decay=0.5)
    assert out2.shape == p.shape
