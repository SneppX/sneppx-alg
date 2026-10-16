"""Segmentation metrics (pure NumPy).

Pixel accuracy, mean IoU, frequency-weighted IoU and the Dice coefficient for
dense per-pixel label predictions.
"""

import numpy as np


def _flat(x) -> np.ndarray:
    return np.asarray(x).ravel().astype(np.int64)


def confusion_matrix(y_true, y_pred, num_classes=None) -> np.ndarray:
    """Pixel-level confusion matrix, shape ``(C, C)``."""
    yt = _flat(y_true)
    yp = _flat(y_pred)
    if num_classes is None:
        num_classes = int(max(yt.max(), yp.max())) + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (yt, yp), 1)
    return cm


def pixel_accuracy(y_true, y_pred) -> float:
    yt = _flat(y_true)
    yp = _flat(y_pred)
    return float(np.mean(yt == yp)) if yt.size else 0.0


def _iou_per_class(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    denom = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, tp / denom, np.nan)


def mean_iou(y_true, y_pred, num_classes=None) -> float:
    """Mean IoU over classes present in either the target or the prediction."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    iou = _iou_per_class(cm)
    valid = ~np.isnan(iou)
    return float(iou[valid].mean()) if valid.any() else 0.0


def frequency_weighted_iou(y_true, y_pred, num_classes=None) -> float:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    iou = _iou_per_class(cm)
    freq = cm.sum(axis=1).astype(np.float64)
    valid = ~np.isnan(iou)
    total = freq[valid].sum()
    if total == 0:
        return 0.0
    return float((iou[valid] * freq[valid]).sum() / total)


def dice_coefficient(y_true, y_pred, num_classes=None) -> float:
    """Mean Dice (F1) over classes, computed from the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, num_classes).astype(np.float64)
    tp = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = np.where(denom > 0, 2 * tp / denom, np.nan)
    valid = ~np.isnan(dice)
    return float(dice[valid].mean()) if valid.any() else 0.0
