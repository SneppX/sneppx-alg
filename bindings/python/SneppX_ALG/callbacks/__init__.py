"""SNEPPX training callbacks — hooks, history, checkpointing, LR scheduling.

from SneppX_ALG import callbacks

cbs = callbacks.CallbackList([
    callbacks.History(),
    callbacks.ModelCheckpoint(monitor="val_loss"),
])
cbs.on_train_begin()
for epoch in range(epochs):
    cbs.on_epoch_end(epoch, {"loss": ..., "val_loss": ...})
cbs.on_train_end()
"""

from . import base, checkpoint, logger, lr_scheduler, progress
from .base import Callback, CallbackList
from .checkpoint import ModelCheckpoint
from .logger import CSVLogger, History, LambdaCallback
from .lr_scheduler import ReduceLROnPlateau, WarmupLR
from .progress import ProgressBar

__all__ = [
    "base",
    "checkpoint",
    "logger",
    "lr_scheduler",
    "progress",
    "Callback",
    "CallbackList",
    "ModelCheckpoint",
    "CSVLogger",
    "History",
    "LambdaCallback",
    "ReduceLROnPlateau",
    "WarmupLR",
    "ProgressBar",
]
