"""SNEPPX signal processing — windows, FFT, filtering, spectral, resampling.

from SneppX_ALG import signal

freqs, psd = signal.welch(x, fs=1000.0)
X = signal.fft.fft(x)          # FFT lives in the ``signal.fft`` submodule
w = signal.hann(64)
"""

from . import fft, filtering, resampling, spectral, windows
from .fft import (
    amplitude_spectrum,
    fft2,
    fftfreq,
    fftshift,
    ifft,
    ifft2,
    ifftshift,
    irfft,
    rfft,
    rfftfreq,
)
from .filtering import (
    design_fir_lowpass,
    exponential_moving_average,
    median_filter,
    moving_average,
)
from .filtering import convolve as convolve
from .filtering import correlate as correlate
from .resampling import decimate, resample, resample_poly
from .spectral import periodogram, welch
from .spectral import spectrogram as spectrogram
from .spectral import stft as stft
from .windows import (
    bartlett,
    blackman,
    boxcar,
    hann,
    hamming,
    kaiser,
)
from .windows import get_window as get_window

__all__ = [
    "windows",
    "fft",
    "filtering",
    "spectral",
    "resampling",
    "boxcar",
    "hann",
    "hamming",
    "blackman",
    "bartlett",
    "kaiser",
    "get_window",
    "ifft",
    "rfft",
    "irfft",
    "fft2",
    "ifft2",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
    "amplitude_spectrum",
    "convolve",
    "correlate",
    "moving_average",
    "exponential_moving_average",
    "median_filter",
    "design_fir_lowpass",
    "periodogram",
    "welch",
    "stft",
    "spectrogram",
    "resample",
    "decimate",
    "resample_poly",
]
