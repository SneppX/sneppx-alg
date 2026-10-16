"""Regression metrics (pure NumPy).

Mirrors scikit-learn's regression scorers: ``mean_squared_error``,
``mean_absolute_error``, ``r2_score`` and related error/skill measures.
"""

import numpy as np


def _flat(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def mean_squared_error(y_true, y_pred) -> float:
    a, b = _flat(y_true), _flat(y_pred)
    return float(np.mean((a - b) ** 2))


def mean_absolute_error(y_true, y_pred) -> float:
    a, b = _flat(y_true), _flat(y_pred)
    return float(np.mean(np.abs(a - b)))


def root_mean_squared_error(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def root_mean_squared_log_error(y_true, y_pred) -> float:
    a, b = _flat(y_true), _flat(y_pred)
    return float(np.sqrt(np.mean((np.log1p(a) - np.log1p(b)) ** 2)))


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    a, b = _flat(y_true), _flat(y_pred)
    denom = np.where(a == 0, 1e-10, np.abs(a))
    return float(np.mean(np.abs(a - b) / denom))


def median_absolute_error(y_true, y_pred) -> float:
    a, b = _flat(y_true), _flat(y_pred)
    return float(np.median(np.abs(a - b)))


def max_error(y_true, y_pred) -> float:
    a, b = _flat(y_true), _flat(y_pred)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def r2_score(y_true, y_pred) -> float:
    a, b = _flat(y_true), _flat(y_pred)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def explained_variance_score(y_true, y_pred) -> float:
    a, b = _flat(y_true), _flat(y_pred)
    var_a = float(np.var(a))
    if var_a == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return 1.0 - float(np.var(a - b)) / var_a


def huber_loss(y_true, y_pred, delta: float = 1.0) -> float:
    a, b = _flat(y_true), _flat(y_pred)
    r = np.abs(a - b)
    quad = np.minimum(r, delta)
    lin = r - quad
    return float(np.mean(0.5 * quad**2 + delta * lin))


def mean_squared_error_metrics(y_true, y_pred) -> dict:
    """Convenience bundle of the most common regression metrics."""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
