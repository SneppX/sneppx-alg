"""Window functions for spectral analysis (pure NumPy)."""

import numpy as np

__all__ = [
    "boxcar",
    "hann",
    "hamming",
    "blackman",
    "bartlett",
    "kaiser",
    "get_window",
]


def _n(n: int) -> int:
    return int(n)


def boxcar(n: int, sym: bool = True) -> np.ndarray:
    return np.ones(_n(n))


def hann(n: int, sym: bool = True) -> np.ndarray:
    return np.hanning(_n(n)) if sym else _periodic(np.hanning, n)


def hamming(n: int, sym: bool = True) -> np.ndarray:
    return np.hamming(_n(n)) if sym else _periodic(np.hamming, n)


def blackman(n: int, sym: bool = True) -> np.ndarray:
    return np.blackman(_n(n)) if sym else _periodic(np.blackman, n)


def bartlett(n: int, sym: bool = True) -> np.ndarray:
    return np.bartlett(_n(n)) if sym else _periodic(np.bartlett, n)


def _periodic(fn, n: int) -> np.ndarray:
    n = _n(n)
    return fn(n + 1)[:n]


def kaiser(n: int, beta: float, sym: bool = True) -> np.ndarray:
    return np.kaiser(_n(n), beta)


def get_window(name: str, n: int, **kwargs) -> np.ndarray:
    """Look up a window by name."""
    table = {
        "boxcar": boxcar,
        "rect": boxcar,
        "hann": hann,
        "hanning": hann,
        "hamming": hamming,
        "blackman": blackman,
        "bartlett": bartlett,
        "kaiser": kaiser,
    }
    if name not in table:
        raise ValueError(f"unknown window {name!r}")
    return table[name](n, **kwargs)
