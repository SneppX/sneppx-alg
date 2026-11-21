import numpy as np

from SneppX_ALG import signal as sig


def test_windows():
    w = sig.hann(5)
    assert np.allclose(w, [0.0, 0.5, 1.0, 0.5, 0.0])
    assert np.allclose(sig.boxcar(3), [1, 1, 1])
    assert sig.hamming(8).shape == (8,)
    assert sig.get_window("hann", 4).shape == (4,)


def test_fft_roundtrip():
    x = np.random.default_rng(0).random(16)
    X = sig.fft.fft(x)
    assert np.allclose(sig.ifft(X).real, x)
    r = sig.fft.rfft(x)
    assert np.allclose(sig.irfft(r, n=16), x)


def test_fftfreq():
    f = sig.fftfreq(4, d=0.1)
    assert np.allclose(f, [0.0, 2.5, -5.0, -2.5])
    fr = sig.rfftfreq(4, d=0.1)
    assert np.allclose(fr, [0.0, 2.5, 5.0])


def test_amplitude_spectrum():
    fs = 100.0
    t = np.arange(1000) / fs
    x = np.sin(2 * np.pi * 10.0 * t)
    freqs, mag = sig.amplitude_spectrum(x, fs)
    peak = freqs[np.argmax(mag)]
    assert abs(peak - 10.0) < 1.0


def test_filtering():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(sig.convolve([1, 2, 3], [0, 1, 0.5]), [0, 1, 2.5, 4, 1.5])
    avg = sig.moving_average(np.ones(5), 3)
    assert np.allclose(avg[1:-1], 1.0)
    ema = sig.exponential_moving_average(np.array([0.0, 1.0, 1.0]), alpha=0.5)
    assert np.allclose(ema, [0.0, 0.5, 0.75])


def test_fir_lowpass():
    h = sig.design_fir_lowpass(0.1, 21)
    assert abs(np.sum(h) - 1.0) < 1e-9
    # low-frequency should pass, high-frequency should be attenuated
    n = 64
    t = np.arange(n)
    low = np.cos(2 * np.pi * 0.02 * t)
    high = np.cos(2 * np.pi * 0.4 * t)
    assert np.sum(np.abs(np.convolve(low, h, "same"))) > np.sum(
        np.abs(np.convolve(high, h, "same"))
    )


def test_periodogram():
    fs = 100.0
    t = np.arange(1024) / fs
    x = np.sin(2 * np.pi * 10.0 * t)
    freqs, psd = sig.periodogram(x, fs=fs)
    assert abs(freqs[np.argmax(psd)] - 10.0) < 1.0


def test_welch_and_spectrogram():
    fs = 100.0
    t = np.arange(1024) / fs
    x = np.sin(2 * np.pi * 10.0 * t)
    freqs, psd = sig.welch(x, fs=fs, nperseg=256)
    assert abs(freqs[np.argmax(psd)] - 10.0) < 2.0
    f2, tt, S = sig.spectrogram(x, fs=fs, nperseg=256)
    assert S.shape[0] == f2.shape[0] and S.shape[1] == tt.shape[0]


def test_resampling():
    x = np.linspace(0.0, 1.0, 10)
    y = sig.resample(x, 20)
    assert y.shape == (20,)
    assert abs(y[0] - 0.0) < 1e-9 and abs(y[-1] - 1.0) < 1e-9
    d = sig.decimate(np.arange(20.0), 2)
    assert d.shape[0] == 10
    p = sig.resample_poly(np.arange(8.0), 2, 1)
    assert p.shape[0] == 16
