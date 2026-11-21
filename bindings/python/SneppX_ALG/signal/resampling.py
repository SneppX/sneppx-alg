"""Resampling helpers (pure NumPy)."""

import numpy as np


def resample(x: np.ndarray, num: int) -> np.ndarray:
    """Resample a signal to ``num`` samples using linear interpolation."""
    x = np.asarray(x, dtype=np.float64)
    num = int(num)
    if x.shape[0] == 0:
        return x
    if x.shape[0] == 1:
        return np.full((num,) + x.shape[1:], x[0])
    old = np.linspace(0.0, 1.0, x.shape[0])
    new = np.linspace(0.0, 1.0, num)
    out = np.empty((num,) + x.shape[1:], dtype=np.float64)
    flat = x.reshape(x.shape[0], -1)
    for i in range(flat.shape[1]):
        out.reshape(num, -1)[:, i] = np.interp(new, old, flat[:, i])
    return out


def decimate(x: np.ndarray, q: int, zero_phase: bool = True) -> np.ndarray:
    """Downsample by an integer factor ``q`` after average (anti-alias) filtering."""
    x = np.asarray(x, dtype=np.float64)
    q = int(q)
    if q < 1:
        raise ValueError("q must be >= 1")
    if q == 1:
        return x.copy()
    kernel = np.ones(q) / q
    smoothed = np.convolve(x, kernel, mode="same")
    return smoothed[::q]


def resample_poly(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """Rational resampling by ``up/down`` via upsample -> filter -> decimate."""
    x = np.asarray(x, dtype=np.float64)
    up, down = int(up), int(down)
    if up < 1 or down < 1:
        raise ValueError("up and down must be >= 1")
    n = x.shape[0] * up
    upsampled = np.zeros(n, dtype=np.float64)
    upsampled[::up] = x
    kernel = np.hamming(max(2 * max(up, down), 8))
    kernel = kernel / np.sum(kernel)
    filtered = np.convolve(upsampled, kernel, mode="same")
    return filtered[::down]
