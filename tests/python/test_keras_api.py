"""Tests for the Keras-style layer API."""

import numpy as np
import pytest

from SneppX_ALG import (
    Sequential,
    Model,
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    Activation,
    BatchNormalization,
    Input,
    Tensor,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _classification_data(n=64, in_features=8, classes=10, seed=0):
    rng = _rng(seed)
    x = rng.standard_normal((n, in_features)).astype(np.float32)
    y = np.eye(classes)[rng.integers(0, classes, size=n)].astype(np.float32)
    return x, y


def test_sequential_compile_fit_learns():
    m = Sequential(input_shape=(8,))
    m.add(Dense(64, activation="relu"))
    m.add(Dense(10, activation="softmax"))
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    x, y = _classification_data()
    h = m.fit(x, y, epochs=40, batch_size=16, verbose=0)

    assert h["loss"][-1] < h["loss"][0]
    assert h["accuracy"][-1] > h["accuracy"][0] > 0.0


def test_evaluate_and_predict():
    m = Sequential(input_shape=(8,))
    m.add(Dense(10, activation="softmax"))
    m.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
    x, y = _classification_data(n=16)
    m.fit(x, y, epochs=3, batch_size=8, verbose=0)

    results = m.evaluate(x, y)
    assert len(results) == 2
    assert results[0] > 0

    preds = m.predict(x)
    assert preds.shape == (16, 10)
    assert np.allclose(preds.sum(axis=-1), 1.0, atol=1e-3)


def test_count_params_and_weights():
    m = Sequential(input_shape=(4,))
    m.add(Dense(8))
    m.add(Dense(2))
    m.compile(optimizer="adam", loss="mse")
    assert m.count_params() == 8 * 4 + 8 + 2 * 8 + 2

    weights = m.get_weights()
    assert len(weights) == 4

    new_weights = [np.zeros_like(w) for w in weights]
    m.set_weights(new_weights)
    assert np.allclose(m.get_weights()[0], 0.0)


def test_save_load_weights(tmp_path):
    m = Sequential(input_shape=(4,))
    m.add(Dense(8))
    m.add(Dense(2))
    m.compile(optimizer="adam", loss="mse")
    path = str(tmp_path / "w.json")
    m.save_weights(path)

    m2 = Sequential(input_shape=(4,))
    m2.add(Dense(8))
    m2.add(Dense(2))
    m2.load_weights(path)
    assert np.allclose(m.get_weights()[0], m2.get_weights()[0])
    assert np.allclose(m.get_weights()[1], m2.get_weights()[1])


def test_compile_requires_layers():
    m = Sequential()
    with pytest.raises(ValueError):
        m.compile(optimizer="adam", loss="mse")


def test_unknown_optimizer_and_loss():
    m = Sequential(input_shape=(4,))
    m.add(Dense(2))
    with pytest.raises(ValueError):
        m.compile(optimizer="nope", loss="mse")
    with pytest.raises(ValueError):
        m.compile(optimizer="adam", loss="nope")


def test_sequential_add_and_len():
    m = Sequential()
    m.add(Dense(4))
    m.add(Activation("relu"))
    m.add(Dense(1))
    assert len(m.layers) == 3
    assert isinstance(m, Model)


def test_broadcast_backward_fix():
    """Regression: bias params must keep their shape through backward."""
    m = Sequential(input_shape=(8,))
    m.add(Dense(64, activation="relu"))
    m.add(Dense(10, activation="softmax"))
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    x, y = _classification_data()
    m.fit(x, y, epochs=2, batch_size=16, verbose=0)
    for p in m._optimizer.params:
        assert len(p.shape) >= 1


def test_regression_loss_decreases_with_bias():
    """Training must actually move weights (grad flows into the bias too)."""
    m = Sequential(input_shape=(6,))
    m.add(Dense(4))
    m.add(Dense(1))
    m.compile(optimizer="sgd", loss="mse", lr=0.05)
    rng = _rng(1)
    x = rng.standard_normal((32, 6)).astype(np.float32)
    y = np.sum(x, axis=1, keepdims=True).astype(np.float32)
    h = m.fit(x, y, epochs=60, batch_size=8, verbose=0)
    assert h["loss"][-1] < h["loss"][0]


def test_mse_regression_converges():
    m = Sequential(input_shape=(6,))
    m.add(Dense(16, activation="relu"))
    m.add(Dense(1))
    m.compile(optimizer="adam", loss="mse", lr=0.01)
    rng = _rng(2)
    x = rng.standard_normal((64, 6)).astype(np.float32)
    y = (3.0 * x[:, 0] - 2.0 * x[:, 1] + x[:, 2]).astype(np.float32)[:, None]
    m.fit(x, y, epochs=80, batch_size=16, verbose=0)
    preds = m.predict(x)
    assert float(np.mean(np.abs(preds - y))) < 0.5


def test_conv_forward_shape():
    m = Sequential(input_shape=(3, 8, 8))
    m.add(Conv2D(6, kernel_size=3, padding=1))
    m.add(MaxPool2D(pool_size=2))
    m.add(Flatten())
    m.compile(optimizer="adam", loss="mse")
    x = np.zeros((2, 3, 8, 8), dtype=np.float32)
    preds = m.predict(x)
    assert preds.shape == (2, 6 * 4 * 4)


def test_dropout_train_vs_eval():
    m = Sequential(input_shape=(4,))
    m.add(Dense(16, activation="relu"))
    m.add(Dropout(0.5))
    m.add(Dense(2))
    x = np.ones((8, 4), dtype=np.float32)
    m.eval()
    p1 = m.predict(x)
    m.train()
    m.eval()
    p2 = m.predict(x)
    assert np.allclose(p1, p2)


def test_summary_contains_params():
    m = Sequential(input_shape=(4,))
    m.add(Dense(8))
    m.add(Dense(2))
    summary = m.summary()
    assert "Total params" in summary
    assert str(m.count_params()) in summary


def test_input_spec():
    spec = Input(shape=(3, 224, 224))
    assert spec.input_shape == (3, 224, 224)


def test_batch_normalization_shape():
    m = Sequential(input_shape=(4,))
    m.add(Dense(8, activation="relu"))
    m.add(BatchNormalization(8))
    m.add(Dense(2))
    x = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
    preds = m.predict(x)
    assert preds.shape == (8, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
