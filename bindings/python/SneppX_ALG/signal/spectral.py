"""Spectral estimation and time-frequency analysis (pure NumPy)."""

from typing import Optional, Tuple

import numpy as np

from .windows import get_window


def periodogram(
    x: np.ndarray, fs: float = 1.0, window: str = "boxcar"
) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided power spectral density estimate: ``(freqs, psd)``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.shape[0]
    w = get_window(window, n)
    xw = x * w
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    scale = 1.0 / (fs * np.sum(w**2))
    psd = (np.abs(X) ** 2) * scale
    if n % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0
    return freqs, psd


def welch(
    x: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch's averaged periodogram: ``(freqs, psd)``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    nperseg = int(min(nperseg, x.shape[0]))
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - int(noverlap)
    if step <= 0:
        step = nperseg
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    acc = np.zeros_like(freqs)
    count = 0
    w = get_window(window, nperseg)
    scale = 1.0 / (fs * np.sum(w**2))
    for start in range(0, x.shape[0] - nperseg + 1, step):
        seg = x[start : start + nperseg] * w
        X = np.fft.rfft(seg)
        psd = (np.abs(X) ** 2) * scale
        if nperseg % 2 == 0:
            psd[1:-1] *= 2.0
        else:
            psd[1:] *= 2.0
        acc += psd
        count += 1
    if count == 0:
        return freqs, acc
    return freqs, acc / count


def spectrogram(
    x: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-time Fourier transform magnitude: ``(freqs, times, Sxx)``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    nperseg = int(min(nperseg, x.shape[0]))
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - int(noverlap)
    if step <= 0:
        step = nperseg
    w = get_window(window, nperseg)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    frames = []
    times = []
    for start in range(0, x.shape[0] - nperseg + 1, step):
        seg = x[start : start + nperseg] * w
        frames.append(np.abs(np.fft.rfft(seg)))
        times.append(start / fs)
    if frames:
        S = np.array(frames).T
    else:
        S = np.zeros((freqs.shape[0], 0))
    return freqs, np.array(times), S


def stft(
    x: np.ndarray,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = "hann",
) -> np.ndarray:
    """Short-time Fourier transform (complex), frames in columns."""
    x = np.asarray(x, dtype=np.float64).ravel()
    nperseg = int(min(nperseg, x.shape[0]))
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - int(noverlap)
    if step <= 0:
        step = nperseg
    w = get_window(window, nperseg)
    frames = []
    for start in range(0, x.shape[0] - nperseg + 1, step):
        frames.append(np.fft.rfft(x[start : start + nperseg] * w))
    return np.array(frames).T if frames else np.zeros((nperseg // 2 + 1, 0), complex)
