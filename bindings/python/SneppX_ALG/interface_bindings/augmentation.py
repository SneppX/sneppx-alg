"""Data augmentation and mixing utilities for training."""

from typing import Optional, Tuple, List, Callable, Any, Union
import numpy as np
import random

from .tensor import Tensor

# =========================================================================
# Image Augmentation
# =========================================================================


def random_resized_crop(
    img: Tensor,
    size: Tuple[int, int],
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (0.75, 1.333),
    interpolation: str = "bilinear",
) -> Tensor:
    """Random resized crop (Inception-style)."""
    x = np.asarray(img.data, dtype=np.float64)
    h, w = x.shape[-2:] if x.ndim >= 2 else (x.shape[-1],)

    # Random area and aspect ratio
    area = h * w
    target_area = random.uniform(*scale) * area
    log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
    aspect_ratio = np.exp(random.uniform(*log_ratio))

    new_w = int(round(np.sqrt(target_area * aspect_ratio)))
    new_h = int(round(np.sqrt(target_area / aspect_ratio)))

    new_w = min(new_w, w)
    new_h = min(new_h, h)

    # Random crop
    i = random.randint(0, h - new_h)
    j = random.randint(0, w - new_w)

    cropped = x[..., i : i + new_h, j : j + new_w]

    # Resize to target size
    out = resize(cropped, size, interpolation)
    return Tensor.from_numpy(out.astype(np.float32))


def resize(
    img: np.ndarray,
    size: Tuple[int, int],
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Resize image using numpy/scipy-style interpolation."""
    # Simplified: use basic scaling
    h, w = img.shape[-2:]
    new_h, new_w = size

    if interpolation == "nearest":
        # Nearest neighbor
        h_indices = (np.arange(new_h) * h / new_h).astype(int)
        w_indices = (np.arange(new_w) * w / new_w).astype(int)
        return img[..., h_indices[:, None], w_indices[None, :]]
    else:
        # Bilinear (simplified)
        h_coords = np.arange(new_h) * h / new_h
        w_coords = np.arange(new_w) * w / new_w

        h0 = np.floor(h_coords).astype(int)
        h1 = np.minimum(h0 + 1, h - 1)
        w0 = np.floor(w_coords).astype(int)
        w1 = np.minimum(w0 + 1, w - 1)

        h_weight = h_coords - h0
        w_weight = w_coords - w0

        # Vectorized bilinear
        if img.ndim == 3:  # C, H, W
            c = img.shape[0]
            out = np.zeros((c, new_h, new_w), dtype=img.dtype)
            for i in range(c):
                v00 = img[i, h0[:, None], w0[None, :]]
                v01 = img[i, h0[:, None], w1[None, :]]
                v10 = img[i, h1[:, None], w0[None, :]]
                v11 = img[i, h1[:, None], w1[None, :]]

                out[i] = (
                    v00 * (1 - h_weight[:, None]) * (1 - w_weight[None, :])
                    + v01 * (1 - h_weight[:, None]) * w_weight[None, :]
                    + v10 * h_weight[:, None] * (1 - w_weight[None, :])
                    + v11 * h_weight[:, None] * w_weight[None, :]
                )
            return out
        else:
            # 2D
            v00 = img[h0[:, None], w0[None, :]]
            v01 = img[h0[:, None], w1[None, :]]
            v10 = img[h1[:, None], w0[None, :]]
            v11 = img[h1[:, None], w1[None, :]]

            return (
                v00 * (1 - h_weight[:, None]) * (1 - w_weight[None, :])
                + v01 * (1 - h_weight[:, None]) * w_weight[None, :]
                + v10 * h_weight[:, None] * (1 - w_weight[None, :])
                + v11 * h_weight[:, None] * w_weight[None, :]
            )


def random_horizontal_flip(img: Tensor, p: float = 0.5) -> Tensor:
    """Random horizontal flip."""
    if random.random() < p:
        x = np.asarray(img.data, dtype=np.float64)
        return Tensor.from_numpy(np.flip(x, axis=-1).astype(np.float32))
    return img


def random_vertical_flip(img: Tensor, p: float = 0.5) -> Tensor:
    """Random vertical flip."""
    if random.random() < p:
        x = np.asarray(img.data, dtype=np.float64)
        return Tensor.from_numpy(np.flip(x, axis=-2).astype(np.float32))
    return img


def random_rotation(
    img: Tensor,
    degrees: Union[float, Tuple[float, float]],
    interpolation: str = "bilinear",
) -> Tensor:
    """Random rotation."""
    if isinstance(degrees, (int, float)):
        angle = random.uniform(-degrees, degrees)
    else:
        angle = random.uniform(degrees[0], degrees[1])

    # Simplified: use np.rot90 for 90-degree increments
    # For arbitrary angles, would need scipy.ndimage.rotate
    k = int(round(angle / 90)) % 4
    x = np.asarray(img.data, dtype=np.float64)
    rotated = np.rot90(x, k=k, axes=(-2, -1))
    return Tensor.from_numpy(rotated.astype(np.float32))


def color_jitter(
    img: Tensor,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
) -> Tensor:
    """Random color jitter."""
    x = np.asarray(img.data, dtype=np.float64)

    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        x = x * factor

    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        mean = np.mean(x, axis=(-2, -1), keepdims=True)
        x = mean + (x - mean) * factor

    # Saturation and hue would require HSV conversion
    # Simplified here

    return Tensor.from_numpy(np.clip(x, 0, 1).astype(np.float32))


def random_grayscale(img: Tensor, p: float = 0.1) -> Tensor:
    """Randomly convert to grayscale."""
    if random.random() < p:
        x = np.asarray(img.data, dtype=np.float64)
        if x.shape[-3] == 3:  # RGB
            gray = (
                0.2989 * x[..., 0, :, :]
                + 0.5870 * x[..., 1, :, :]
                + 0.1140 * x[..., 2, :, :]
            )
            x = np.stack([gray, gray, gray], axis=-3)
        return Tensor.from_numpy(x.astype(np.float32))
    return img


def gaussian_blur(
    img: Tensor, kernel_size: int = 3, sigma: Tuple[float, float] = (0.1, 2.0)
) -> Tensor:
    """Gaussian blur."""
    x = np.asarray(img.data, dtype=np.float64)
    # Simplified: use scipy.ndimage.gaussian_filter if available
    # Simplified here
    return img


def solarize(img: Tensor, threshold: float = 0.5) -> Tensor:
    """Solarize image."""
    x = np.asarray(img.data, dtype=np.float64)
    x = np.where(x < threshold, x, 1.0 - x)
    return Tensor.from_numpy(x.astype(np.float32))


def posterize(img: Tensor, bits: int) -> Tensor:
    """Posterize image."""
    x = np.asarray(img.data, dtype=np.float64)
    levels = 2**bits
    x = np.round(x * (levels - 1)) / (levels - 1)
    return Tensor.from_numpy(x.astype(np.float32))


def auto_contrast(img: Tensor) -> Tensor:
    """Auto contrast."""
    x = np.asarray(img.data, dtype=np.float64)
    for c in range(x.shape[-3]):
        ch = x[..., c, :, :]
        lo = np.min(ch)
        hi = np.max(ch)
        if hi > lo:
            x[..., c, :, :] = (ch - lo) / (hi - lo)
    return Tensor.from_numpy(x.astype(np.float32))


def equalize(img: Tensor) -> Tensor:
    """Histogram equalization."""
    x = np.asarray(img.data, dtype=np.float64)
    for c in range(x.shape[-3]):
        ch = x[..., c, :, :]
        hist, bins = np.histogram(ch.flatten(), 256, [0, 1])
        cdf = hist.cumsum()
        cdf = 255 * cdf / cdf[-1]
        ch = np.interp(ch.flatten(), bins[:-1], cdf).reshape(ch.shape)
        x[..., c, :, :] = ch / 255.0
    return Tensor.from_numpy(x.astype(np.float32))


def invert(img: Tensor) -> Tensor:
    """Invert colors."""
    x = np.asarray(img.data, dtype=np.float64)
    return Tensor.from_numpy((1.0 - x).astype(np.float32))


def sharpness(img: Tensor, factor: float = 1.0) -> Tensor:
    """Adjust sharpness."""
    if factor == 1.0:
        return img
    x = np.asarray(img.data, dtype=np.float64)
    # Simplified unsharp mask
    blurred = gaussian_blur(img, kernel_size=3)
    out = img + (img - blurred) * factor
    return Tensor.from_numpy(np.clip(out.data, 0, 1).astype(np.float32))


# =========================================================================
# Mixup / CutMix / AugMix
# =========================================================================


def mixup(
    batch: Tensor,
    targets: Tensor,
    alpha: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, float]:
    """Mixup data augmentation."""
    batch_size = batch.shape[0]
    lam = np.random.beta(alpha, alpha)

    index = np.random.permutation(batch_size)

    mixed_batch = lam * batch + (1 - lam) * batch[index]
    targets_a = targets
    targets_b = targets[index]

    return mixed_batch, targets_a, targets_b, lam


def cutmix(
    batch: Tensor,
    targets: Tensor,
    alpha: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, float]:
    """CutMix data augmentation."""
    batch_size = batch.shape[0]
    lam = np.random.beta(alpha, alpha)

    index = np.random.permutation(batch_size)

    _, _, h, w = batch.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    batch[:, :, bby1:bby2, bbx1:bbx2] = batch[index, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

    return batch, targets, targets[index], lam


def augmix(
    batch: Tensor,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
) -> Tensor:
    """AugMix augmentation."""
    # Simplified AugMix
    # Full implementation would apply chains of augmentations
    return batch


# =========================================================================
# Cutout / Random Erasing
# =========================================================================


def cutout(
    img: Tensor,
    num_holes: int = 1,
    hole_size: Tuple[int, int] = (16, 16),
) -> Tensor:
    """CutOut / Random Erasing."""
    x = np.asarray(img.data, dtype=np.float64).copy()
    _, h, w = x.shape[-3:]

    hole_h, hole_w = hole_size

    for _ in range(num_holes):
        y = np.random.randint(0, h - hole_h + 1)
        x_ = np.random.randint(0, w - hole_w + 1)
        x[..., y : y + hole_h, x_ : x_ + hole_w] = 0

    return Tensor.from_numpy(x.astype(np.float32))


def random_erasing(
    img: Tensor,
    p: float = 0.5,
    scale: Tuple[float, float] = (0.02, 0.33),
    ratio: Tuple[float, float] = (0.3, 3.3),
    value: Union[float, Tuple[float, float, float]] = 0,
    inplace: bool = False,
) -> Tensor:
    """Random Erasing data augmentation."""
    if random.random() > p:
        return img

    x = np.asarray(img.data, dtype=np.float64)
    if not inplace:
        x = x.copy()

    _, h, w = x.shape[-3:]
    area = h * w

    for attempt in range(10):
        target_area = random.uniform(*scale) * area
        log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
        aspect_ratio = np.exp(random.uniform(*log_ratio))

        h_ = int(round(np.sqrt(target_area * aspect_ratio)))
        w_ = int(round(np.sqrt(target_area / aspect_ratio)))

        if h_ < h and w_ < w:
            i = random.randint(0, h - h_)
            j = random.randint(0, w - w_)

            if isinstance(value, (int, float)):
                v = value
            else:
                v = np.array(value, dtype=np.float64).reshape(-1, 1, 1)

            x[..., i : i + h_, j : j + w_] = v
            break

    return Tensor.from_numpy(x.astype(np.float32))


# =========================================================================
# Text Augmentation
# =========================================================================


def random_token_mask(
    input_ids: Tensor,
    mask_token_id: int,
    mask_prob: float = 0.15,
    replace_prob: float = 0.8,
    random_prob: float = 0.1,
    pad_token_id: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Random token masking for MLM (BERT-style)."""
    input_ids_np = np.asarray(input_ids.data, dtype=np.int64).copy()
    labels = np.full_like(input_ids_np, -100)

    batch, seq_len = input_ids_np.shape

    # Create mask
    prob = np.random.random((batch, seq_len))
    mask = (prob < mask_prob) & (input_ids_np != 0) & (input_ids_np != pad_token_id)

    # Replace with [MASK]
    rand = np.random.random((batch, seq_len))
    mask_token = mask & (rand < replace_prob)
    input_ids_np[mask_token] = mask_token_id

    # Replace with random token
    random_token = mask & (~mask_token) & (rand < replace_prob + random_prob)
    if np.any(random_token):
        vocab_size = 30522  # default BERT vocab
        random_tokens = np.random.randint(0, vocab_size, size=np.sum(random_token))
        input_ids_np[random_token] = random_tokens

    # Keep original for remaining
    labels[mask] = input_ids.data[mask]

    return Tensor.from_numpy(input_ids_np.astype(np.int64)), Tensor.from_numpy(
        labels.astype(np.int64)
    )


def random_word_dropout(
    input_ids: Tensor,
    dropout_prob: float = 0.1,
    pad_token_id: int = 0,
) -> Tensor:
    """Randomly drop words (replace with pad)."""
    input_ids_np = np.asarray(input_ids.data, dtype=np.int64).copy()
    mask = (np.random.random(input_ids_np.shape) < dropout_prob) & (
        input_ids_np != pad_token_id
    )
    input_ids_np[mask] = pad_token_id
    return Tensor.from_numpy(input_ids_np.astype(np.int64))


def random_word_shuffle(
    input_ids: Tensor,
    shuffle_prob: float = 0.1,
    pad_token_id: int = 0,
) -> Tensor:
    """Randomly shuffle words."""
    input_ids_np = np.asarray(input_ids.data, dtype=np.int64).copy()
    batch, seq_len = input_ids_np.shape

    for b in range(batch):
        # Find non-pad tokens
        non_pad = np.where(input_ids_np[b] != pad_token_id)[0]
        if len(non_pad) > 1:
            # Shuffle a subset
            num_shuffle = int(len(non_pad) * shuffle_prob)
            if num_shuffle > 1:
                shuffle_idx = np.random.choice(non_pad, num_shuffle, replace=False)
                shuffled = input_ids_np[b, shuffle_idx].copy()
                np.random.shuffle(shuffled)
                input_ids_np[b, shuffle_idx] = shuffled

    return Tensor.from_numpy(input_ids_np.astype(np.int64))


def token_cutout(
    input_ids: Tensor,
    cutout_prob: float = 0.1,
    cutout_len: int = 3,
    pad_token_id: int = 0,
) -> Tensor:
    """Cutout contiguous spans of tokens."""
    input_ids_np = np.asarray(input_ids.data, dtype=np.int64).copy()
    batch, seq_len = input_ids_np.shape

    for b in range(batch):
        non_pad = np.where(input_ids_np[b] != pad_token_id)[0]
        if len(non_pad) == 0:
            continue

        num_cuts = int(len(non_pad) * cutout_prob / cutout_len)
        for _ in range(num_cuts):
            if len(non_pad) < cutout_len:
                break
            start = np.random.randint(0, len(non_pad) - cutout_len + 1)
            cut_indices = non_pad[start : start + cutout_len]
            input_ids_np[b, cut_indices] = pad_token_id
            # Update non_pad
            non_pad = non_pad[~np.isin(non_pad, cut_indices)]

    return Tensor.from_numpy(input_ids_np.astype(np.int64))


# =========================================================================
# Audio Augmentation
# =========================================================================


def time_stretch(audio: Tensor, rate: float) -> Tensor:
    """Time stretch audio."""
    x = np.asarray(audio.data, dtype=np.float64)
    # Simplified: would use librosa.effects.time_stretch
    return audio


def pitch_shift(audio: Tensor, n_steps: float, sample_rate: int = 16000) -> Tensor:
    """Pitch shift audio."""
    # Would use librosa.effects.pitch_shift
    return audio


def time_mask(audio: Tensor, mask_param: int = 50, p: float = 0.5) -> Tensor:
    """SpecAugment time masking."""
    if random.random() > p:
        return audio
    x = np.asarray(audio.data, dtype=np.float64).copy()
    _, time = x.shape[-2:]
    t = min(mask_param, time // 2)
    if t == 0:
        return audio
    t0 = random.randint(0, time - t)
    x[..., t0 : t0 + t] = 0
    return Tensor.from_numpy(x.astype(np.float32))


def freq_mask(audio: Tensor, mask_param: int = 10, p: float = 0.5) -> Tensor:
    """SpecAugment frequency masking."""
    if random.random() > p:
        return audio
    x = np.asarray(audio.data, dtype=np.float64).copy()
    _, freq = x.shape[-2:]
    f = min(mask_param, freq // 2)
    if f == 0:
        return audio
    f0 = random.randint(0, freq - f)
    x[..., f0 : f0 + f, :] = 0
    return Tensor.from_numpy(x.astype(np.float32))


# =========================================================================
# Augmentation Pipeline
# =========================================================================


class Compose:
    """Compose multiple augmentations."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        for t in self.transforms:
            args = t(*args, **kwargs) if not isinstance(args, tuple) else t(*args)
        return args


class RandomApply:
    """Randomly apply a transform with probability p."""

    def __init__(self, transform: Callable, p: float = 0.5):
        self.transform = transform
        self.p = p

    def __call__(self, *args, **kwargs):
        if random.random() < self.p:
            return self.transform(*args, **kwargs)
        return args[0] if len(args) == 1 else args


class RandomChoice:
    """Randomly choose one transform from a list."""

    def __init__(self, transforms: List[Callable], p: Optional[List[float]] = None):
        self.transforms = transforms
        self.p = p

    def __call__(self, *args, **kwargs):
        t = random.choices(self.transforms, weights=self.p, k=1)[0]
        return t(*args, **kwargs)


class RandomOrder:
    """Apply transforms in random order."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            args = (
                self.transforms[i](*args, **kwargs)
                if not isinstance(args, tuple)
                else self.transforms[i](*args)
            )
        return args


# Predefined augmentation policies
IMAGENET_TRAIN_TRANSFORMS = Compose(
    [
        lambda x: random_resized_crop(x, (224, 224)),
        lambda x: random_horizontal_flip(x),
        lambda x: color_jitter(x, brightness=0.4, contrast=0.4, saturation=0.4),
        lambda x: random_grayscale(x, p=0.2),
        lambda x: random_horizontal_flip(x),
    ]
)

IMAGENET_EVAL_TRANSFORMS = Compose(
    [
        lambda x: resize(x, (256, 256)),
        lambda x: center_crop(x, (224, 224)),
    ]
)

CIFAR10_TRAIN_TRANSFORMS = Compose(
    [
        lambda x: random_crop(x, (32, 32), padding=4),
        lambda x: random_horizontal_flip(x),
    ]
)

CIFAR10_EVAL_TRANSFORMS = Compose([])

# =========================================================================
# Helper functions
# =========================================================================


def center_crop(img: Tensor, size: Tuple[int, int]) -> Tensor:
    """Center crop."""
    x = np.asarray(img.data, dtype=np.float64)
    h, w = x.shape[-2:]
    new_h, new_w = size
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return Tensor.from_numpy(
        x[..., top : top + new_h, left : left + new_w].astype(np.float32)
    )


def random_crop(
    img: Tensor,
    size: Tuple[int, int],
    padding: int = 0,
    pad_if_needed: bool = False,
) -> Tensor:
    """Random crop with optional padding."""
    x = np.asarray(img.data, dtype=np.float64)

    if padding > 0:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode="constant")

    h, w = x.shape[-2:]
    new_h, new_w = size

    if h < new_h or w < new_w:
        if pad_if_needed:
            pad_h = max(0, new_h - h)
            pad_w = max(0, new_w - w)
            x = np.pad(
                x,
                (
                    (0, 0),
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                ),
                mode="constant",
            )
            h, w = x.shape[-2:]
        else:
            return Tensor.from_numpy(x.astype(np.float32))

    top = np.random.randint(0, h - new_h + 1)
    left = np.random.randint(0, w - new_w + 1)
    return Tensor.from_numpy(
        x[..., top : top + new_h, left : left + new_w].astype(np.float32)
    )


def five_crop(img: Tensor, size: Tuple[int, int]) -> List[Tensor]:
    """Five crop augmentation (4 corners + center)."""
    x = np.asarray(img.data, dtype=np.float64)
    h, w = x.shape[-2:]
    crop_h, crop_w = size

    crops = []
    # Top-left, top-right, bottom-left, bottom-right, center
    positions = [
        (0, 0),
        (0, w - crop_w),
        (h - crop_h, 0),
        (h - crop_h, w - crop_w),
        ((h - crop_h) // 2, (w - crop_w) // 2),
    ]

    for top, left in positions:
        crops.append(
            Tensor.from_numpy(
                x[..., top : top + crop_h, left : left + crop_w].astype(np.float32)
            )
        )

    return crops


def ten_crop(img: Tensor, size: Tuple[int, int]) -> List[Tensor]:
    """Ten crop (5 crops + 5 flipped)."""
    crops = five_crop(img, size)
    flipped = [random_horizontal_flip(c, p=1.0) for c in crops]
    return crops + flipped
