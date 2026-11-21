"""SNEPPX regularization — penalties, spectral norm, label smoothing,
early stopping, mixup and weight decay (pure NumPy).

    from SneppX_ALG import regularization

    reg = regularization.l1_penalty(w) + regularization.l2_penalty(w)
    smooth = regularization.smooth_labels(y, num_classes=10, eps=0.1)
"""

from . import early_stopping, label_smoothing, mixup, penalties, spectral, weight_decay
from .early_stopping import EarlyStopping, WeightAveraging
from .label_smoothing import confidence_penalty, smooth_labels, smooth_one_hot
from .mixup import cutmix, mixup, mixup_loss
from .penalties import (
    elastic_net,
    group_lasso,
    l1_penalty,
    l2_penalty,
    orthogonality_penalty,
    proximity_penalty,
    total_variation,
)
from .spectral import (
    power_iteration,
    spectral_normalize,
    spectral_norm,
    spectral_penalty,
)
from .weight_decay import (
    add_l2_grad,
    apply_decoupled_weight_decay,
    apply_weight_decay,
)

__all__ = [
    "penalties",
    "spectral",
    "label_smoothing",
    "early_stopping",
    "mixup",
    "weight_decay",
    "l1_penalty",
    "l2_penalty",
    "elastic_net",
    "group_lasso",
    "total_variation",
    "orthogonality_penalty",
    "proximity_penalty",
    "power_iteration",
    "spectral_norm",
    "spectral_normalize",
    "spectral_penalty",
    "smooth_one_hot",
    "smooth_labels",
    "confidence_penalty",
    "EarlyStopping",
    "WeightAveraging",
    "mixup",
    "cutmix",
    "mixup_loss",
    "add_l2_grad",
    "apply_weight_decay",
    "apply_decoupled_weight_decay",
]
