"""Tests for Data Augmentation."""

import numpy as np
from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings.augmentation import (
    random_resized_crop, random_horizontal_flip, random_vertical_flip,
    random_rotation, color_jitter, random_grayscale,
    gaussian_blur, solarize, posterize, equalize,
    mixup, cutmix, cutout, random_erasing,
    Compose, RandomApply, RandomChoice,
    IMAGENET_TRAIN_TRANSFORMS, IMAGENET_EVAL_TRANSFORMS,
)


def _make_img(c=3, h=32, w=32):
    return Tensor.from_numpy(np.random.rand(c, h, w).astype(np.float32))


def test_random_resized_crop():
    img = _make_img(3, 64, 64)
    out = random_resized_crop(img, (32, 32))
    assert out.shape == (3, 32, 32)
    assert isinstance(out, Tensor)


def test_random_horizontal_flip():
    img = _make_img(3, 16, 16)
    out = random_horizontal_flip(img, p=1.0)
    assert out.shape == (3, 16, 16)


def test_random_vertical_flip():
    img = _make_img(3, 16, 16)
    out = random_vertical_flip(img, p=1.0)
    assert out.shape == (3, 16, 16)


def test_random_rotation():
    img = _make_img(3, 16, 16)
    out = random_rotation(img, degrees=30)
    assert out.shape == (3, 16, 16)


def test_color_jitter():
    img = _make_img(3, 16, 16)
    out = color_jitter(img, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
    assert out.shape == (3, 16, 16)


def test_random_grayscale():
    img = _make_img(3, 16, 16)
    out = random_grayscale(img, p=1.0)
    assert out.shape == (3, 16, 16)


def test_gaussian_blur():
    img = _make_img(3, 16, 16)
    out = gaussian_blur(img, kernel_size=5)
    assert out.shape == (3, 16, 16)


def test_solarize():
    img = Tensor.ones((3, 8, 8)) * 0.5
    out = solarize(img, threshold=0.3)
    assert out.shape == (3, 8, 8)


def test_posterize():
    img = Tensor.ones((3, 8, 8)) * 0.5
    out = posterize(img, bits=4)
    assert out.shape == (3, 8, 8)


def test_equalize():
    img = _make_img(3, 8, 8)
    out = equalize(img)
    assert out.shape == (3, 8, 8)


def test_mixup():
    batch = Tensor.ones((4, 3, 8))
    targets = Tensor.from_numpy(np.array([0, 1, 0, 1]))
    mixed_batch, targets_a, targets_b, lam = mixup(batch, targets, alpha=1.0)
    assert mixed_batch.shape == (4, 3, 8)
    assert 0.0 < float(lam) < 1.0


def test_cutmix():
    batch = Tensor.ones((4, 3, 16, 16))
    targets = Tensor.from_numpy(np.array([0, 1, 0, 1]))
    mixed_batch, targets_a, targets_b, lam = cutmix(batch, targets, alpha=1.0)
    assert mixed_batch.shape == (4, 3, 16, 16)
    assert 0.0 < float(lam) < 1.0


def test_cutout():
    img = _make_img(3, 32, 32)
    out = cutout(img, num_holes=1, hole_size=(8, 8))
    assert out.shape == (3, 32, 32)


def test_compose_single():
    transforms = Compose([lambda x: x * 2.0])
    img = _make_img(3, 16, 16)
    out = transforms(img)
    assert out.shape == (3, 16, 16)
    transform = RandomApply(lambda x: x * 0.0, p=1.0)
    img = _make_img(3, 16, 16)
    out = transform(img)
    assert np.allclose(out.data, 0.0)


def test_random_choice():
    transform = RandomChoice([lambda x: x * 2.0, lambda x: x * 3.0])
    img = _make_img(3, 16, 16)
    out = transform(img)
    assert out.shape == (3, 16, 16)


def test_imagenet_transforms_exist():
    assert callable(IMAGENET_TRAIN_TRANSFORMS)
    assert callable(IMAGENET_EVAL_TRANSFORMS)


if __name__ == "__main__":
    import sys
    locals_ = locals().copy()
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  PASS {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
