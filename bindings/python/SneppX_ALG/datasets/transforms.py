"""Array transforms (pure NumPy) — normalization, rescaling, flips, noise."""

from typing import Optional, Tuple

import numpy as np


class Compose:
    """Chain several transforms: ``Compose(t1, t2)(x) == t2(t1(x))``."""

    def __init__(self, *transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class ToFloat32:
    """Cast an array to ``float32``."""

    def __call__(self, x):
        return np.asarray(x, dtype=np.float32)


class Rescale:
    """Multiply by a constant ``scale``."""

    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)

    def __call__(self, x):
        return np.asarray(x) * self.scale


class Normalize:
    """Per-channel/feature ``(x - mean) / std``.

    ``mean``/``std`` may be scalars or broadcastable arrays. For image-like
    arrays laid out ``(..., C)`` the values are broadcast over the last axis.
    """

    def __init__(self, mean, std, channel_last: bool = True):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.channel_last = channel_last

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        mean, std = self.mean, self.std
        if self.channel_last and x.ndim >= 1 and self.mean.ndim == 1:
            mean = self.mean.reshape(1, -1)
            std = self.std.reshape(1, -1)
        return (x - mean) / np.where(std == 0, 1.0, std)


class RandomHorizontalFlip:
    """Flip an array along ``axis`` (default: last) with probability ``p``.

    For channel-last images ``(..., H, W, C)`` pass ``axis=-2``; for plain
    ``(..., H, W)`` arrays the default ``axis=-1`` flips the width.
    """

    def __init__(self, p: float = 0.5, seed: Optional[int] = None, axis: int = -1):
        self.p = float(p)
        self.axis = int(axis)
        self.rng = np.random.default_rng(seed)

    def __call__(self, x):
        x = np.asarray(x)
        if self.rng.random() < self.p:
            return np.flip(x, axis=self.axis).copy()
        return x


class RandomCrop:
    """Random crop of spatial size ``size`` from the leading spatial axes."""

    def __init__(self, size: Tuple[int, int], seed: Optional[int] = None):
        self.size = (int(size[0]), int(size[1]))
        self.rng = np.random.default_rng(seed)

    def __call__(self, x):
        x = np.asarray(x)
        h, w = x.shape[-2], x.shape[-1]
        ch, cw = self.size
        if ch > h or cw > w:
            raise ValueError("crop size larger than input")
        top = self.rng.integers(0, h - ch + 1)
        left = self.rng.integers(0, w - cw + 1)
        return x[..., top : top + ch, left : left + cw].copy()


class GaussianNoise:
    """Add zero-mean Gaussian noise with standard deviation ``std``."""

    def __init__(self, std: float = 0.01, seed: Optional[int] = None):
        self.std = float(std)
        self.rng = np.random.default_rng(seed)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x + self.rng.normal(0.0, self.std, size=x.shape)
