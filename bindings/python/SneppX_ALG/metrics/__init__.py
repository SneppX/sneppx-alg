"""SNEPPX metrics — evaluation metrics for classification, regression,
detection, segmentation, retrieval/ranking and sequence generation.

Pure-NumPy implementations with a scikit-learn-compatible API surface::

    from SneppX_ALG import metrics

    acc = metrics.accuracy_score(y_true, y_pred)
    m = metrics.mean_squared_error(y_true, y_pred)

Every metric is also importable from its submodule, e.g.
``from SneppX_ALG.metrics.classification import f1_score``.
"""

from . import classification, detection, ranking, regression, segmentation, seq2seq
from .classification import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from .detection import (
    box_iou,
    iou_matrix,
    mean_average_precision,
    nms,
    voc_average_precision,
)
from .ranking import (
    average_precision_at_k,
    dcg_at_k,
    hit_rate_at_k,
    mean_average_precision_at_k,
    mean_ndcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .regression import (
    explained_variance_score,
    huber_loss,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_error_metrics,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)
from .segmentation import (
    dice_coefficient,
    frequency_weighted_iou,
    mean_iou,
    pixel_accuracy,
)
from .seq2seq import (
    character_error_rate,
    chrf,
    corpus_bleu,
    edit_distance,
    exact_match,
    rouge_l,
    rouge_n,
    sentence_bleu,
    word_error_rate,
)

__all__ = [
    # submodules
    "classification",
    "regression",
    "detection",
    "segmentation",
    "ranking",
    "seq2seq",
    # classification
    "confusion_matrix",
    "accuracy_score",
    "top_k_accuracy_score",
    "precision_recall_fscore_support",
    "precision_score",
    "recall_score",
    "f1_score",
    "roc_auc_score",
    "average_precision_score",
    "matthews_corrcoef",
    "balanced_accuracy_score",
    "cohen_kappa_score",
    # regression
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "root_mean_squared_log_error",
    "mean_absolute_percentage_error",
    "median_absolute_error",
    "max_error",
    "r2_score",
    "explained_variance_score",
    "huber_loss",
    "mean_squared_error_metrics",
    # detection
    "box_iou",
    "iou_matrix",
    "nms",
    "voc_average_precision",
    "mean_average_precision",
    # segmentation
    "pixel_accuracy",
    "mean_iou",
    "frequency_weighted_iou",
    "dice_coefficient",
    # ranking
    "precision_at_k",
    "recall_at_k",
    "hit_rate_at_k",
    "average_precision_at_k",
    "mean_average_precision_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "dcg_at_k",
    "ndcg_at_k",
    "mean_ndcg_at_k",
    # seq2seq
    "edit_distance",
    "sentence_bleu",
    "corpus_bleu",
    "rouge_n",
    "rouge_l",
    "word_error_rate",
    "character_error_rate",
    "chrf",
    "exact_match",
]
