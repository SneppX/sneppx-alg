"""Time-domain filtering and convolution (pure NumPy)."""

from typing import Optional

import numpy as np


def convolve(a: np.ndarray, v: np.ndarray, mode: str = "full") -> np.ndarray:
    """Discrete convolution."""
    return np.convolve(
        np.asarray(a, dtype=np.float64), np.asarray(v, dtype=np.float64), mode
    )


def correlate(a: np.ndarray, v: np.ndarray, mode: str = "full") -> np.ndarray:
    """Cross-correlation."""
    return np.correlate(
        np.asarray(a, dtype=np.float64), np.asarray(v, dtype=np.float64), mode
    )


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Causal-centred moving average with edge handling (same length as input)."""
    x = np.asarray(x, dtype=np.float64)
    if window <= 0:
        raise ValueError("window must be positive")
    kernel = np.ones(int(window)) / float(window)
    return np.convolve(x, kernel, mode="same")


def exponential_moving_average(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially weighted moving average with smoothing ``alpha`` in (0, 1]."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    acc = x[0]
    out[0] = acc
    for i in range(1, x.shape[0]):
        acc = alpha * x[i] + (1.0 - alpha) * acc
        out[i] = acc
    return out


def median_filter(x: np.ndarray, size: int) -> np.ndarray:
    """Sliding median filter (edge-padded)."""
    x = np.asarray(x, dtype=np.float64)
    size = int(size)
    half = size // 2
    padded = np.pad(x, half, mode="edge")
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = np.median(padded[i : i + size])
    return out


def fir_filter(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply a causal FIR filter (``taps``) to ``x`` (same length, zero-padded)."""
    x = np.asarray(x, dtype=np.float64)
    taps = np.asarray(taps, dtype=np.float64)
    return np.convolve(x, taps, mode="same")


def design_fir_lowpass(cutoff: float, numtaps: int, window=None) -> np.ndarray:
    """Windowed-sinc low-pass FIR filter.

    ``cutoff`` is the normalized cutoff in cycles/sample (0 < cutoff < 0.5).
    """
    if not 0.0 < cutoff < 0.5:
        raise ValueError("cutoff must be in (0, 0.5)")
    n = int(numtaps)
    n = max(3, n)
    if n % 2 == 0:
        n += 1
    m = (n - 1) / 2.0
    k = np.arange(n) - m
    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * k)
    if window is None:
        window = np.hanning(n)
    elif callable(window):
        window = window(n)
    h *= np.asarray(window)
    h /= np.sum(h)
    return h
