"""FFT helpers (thin, documented wrappers over NumPy)."""

from typing import Optional, Tuple

import numpy as np


def fft(x: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    """Complex discrete Fourier transform."""
    return np.fft.fft(np.asarray(x), n=n)


def ifft(X: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    """Inverse complex DFT."""
    return np.fft.ifft(np.asarray(X), n=n)


def rfft(x: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    """DFT of a real signal (non-negative frequencies only)."""
    return np.fft.rfft(np.asarray(x), n=n)


def irfft(X: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    """Inverse of :func:`rfft`."""
    return np.fft.irfft(np.asarray(X), n=n)


def fft2(x: np.ndarray) -> np.ndarray:
    """2-D complex DFT."""
    return np.fft.fft2(np.asarray(x))


def ifft2(X: np.ndarray) -> np.ndarray:
    """2-D inverse complex DFT."""
    return np.fft.ifft2(np.asarray(X))


def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """Sample frequencies for an ``n``-point transform with spacing ``d``."""
    return np.fft.fftfreq(int(n), d)


def rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """Sample frequencies for :func:`rfft`."""
    return np.fft.rfftfreq(int(n), d)


def fftshift(x: np.ndarray) -> np.ndarray:
    """Shift the zero-frequency component to the centre."""
    return np.fft.fftshift(np.asarray(x))


def ifftshift(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`fftshift`."""
    return np.fft.ifftshift(np.asarray(x))


def amplitude_spectrum(x: np.ndarray, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum: ``(freqs, amplitudes)``."""
    x = np.asarray(x)
    n = x.shape[-1]
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    mag = np.abs(X) / n
    return freqs, mag
