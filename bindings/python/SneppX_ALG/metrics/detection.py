"""Object-detection metrics (pure NumPy).

Boxes are corner-form ``[x1, y1, x2, y2]``. Implements IoU, greedy
non-maximum suppression, and VOC-style average precision / mAP.
"""

from typing import List, Sequence

import numpy as np


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def box_iou(box_a, box_b) -> float:
    """IoU between two corner-form boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_area(box_a) + box_area(box_b) - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(boxes_a, boxes_b) -> np.ndarray:
    """IoU for every pair of boxes from two sets -> ``(len(a), len(b))``."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def nms(boxes, scores, iou_threshold: float = 0.5) -> List[int]:
    """Greedy non-maximum suppression; returns kept indices sorted by score."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return keep


def voc_average_precision(recall, precision) -> float:
    """All-point interpolated area under a precision-recall curve."""
    r = np.asarray(recall, dtype=np.float64)
    p = np.asarray(precision, dtype=np.float64)
    mrec = np.concatenate(([0.0], r, [1.0]))
    mpre = np.concatenate(([0.0], p, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def mean_average_precision(
    pred_boxes,
    pred_scores,
    pred_labels,
    gt_boxes,
    gt_labels,
    num_classes: int,
    iou_threshold: float = 0.5,
) -> float:
    """VOC-style mAP over ``num_classes`` (per-image greedy matching)."""
    n_images = len(pred_boxes)
    aps: List[float] = []
    for c in range(1, num_classes + 1):
        all_tp: List[float] = []
        all_conf: List[float] = []
        n_gt = 0
        for i in range(n_images):
            gt = np.asarray(gt_boxes[i], dtype=np.float64).reshape(-1, 4)
            gl = np.asarray(gt_labels[i]).ravel()
            cls_gt = gt[gl == c]
            n_gt += cls_gt.shape[0]
            pb = np.asarray(pred_boxes[i], dtype=np.float64).reshape(-1, 4)
            pl = np.asarray(pred_labels[i]).ravel()
            ps = np.asarray(pred_scores[i], dtype=np.float64).ravel()
            sel = pl == c
            pb, ps = pb[sel], ps[sel]
            order = np.argsort(-ps)
            pb, ps = pb[order], ps[order]
            matched = np.zeros(cls_gt.shape[0], dtype=bool)
            for b, s in zip(pb, ps):
                all_conf.append(float(s))
                best_iou, best_j = 0.0, -1
                for j in range(cls_gt.shape[0]):
                    if matched[j]:
                        continue
                    v = box_iou(b, cls_gt[j])
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= iou_threshold and best_j >= 0:
                    matched[best_j] = True
                    all_tp.append(1.0)
                else:
                    all_tp.append(0.0)
        if n_gt == 0:
            continue
        if not all_tp:
            aps.append(0.0)
            continue
        conf = np.asarray(all_conf)
        order = np.argsort(-conf)
        tp = np.asarray(all_tp)[order]
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(1 - tp)
        recall = tp_cum / n_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        aps.append(voc_average_precision(recall, precision))
    return float(np.mean(aps)) if aps else 0.0
