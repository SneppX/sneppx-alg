"""Callback base classes and a dispatching container (pure Python)."""

from typing import List, Optional


class Callback:
    """Base class for training-loop callbacks.

    Hook methods are intentionally no-ops; subclasses override the ones they
    need. ``logs`` is a plain ``dict`` of current metrics.
    """

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, logs: Optional[dict] = None):
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        pass

    def on_train_end(self, logs: Optional[dict] = None):
        pass


class CallbackList:
    """Dispatch a hook call to an ordered list of callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks: List[Callback] = list(callbacks) if callbacks else []

    def append(self, callback: Callback):
        self.callbacks.append(callback)

    def __len__(self) -> int:
        return len(self.callbacks)

    def __iter__(self):
        return iter(self.callbacks)

    def set_trainer(self, trainer):
        for cb in self.callbacks:
            cb.set_trainer(trainer)

    def _dispatch(self, method: str, *args):
        for cb in self.callbacks:
            getattr(cb, method)(*args)

    def on_train_begin(self, logs=None):
        self._dispatch("on_train_begin", logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._dispatch("on_epoch_begin", epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self._dispatch("on_epoch_end", epoch, logs)

    def on_train_end(self, logs=None):
        self._dispatch("on_train_end", logs)
