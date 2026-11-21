"""DistributedWrapper — DDP-style module wrapper with gradient sync and device placement."""

from typing import Optional, List
from .tensor import Tensor
from .nn import Module
from .distributed import (
    get_world_size,
    get_rank,
    all_reduce,
    barrier,
    _global_context,
)


class DistributedWrapper(Module):
    """Wraps a Module for distributed data-parallel training.

    Handles device placement, gradient all-reduce after backward,
    and optional batch-norm synchronization.

    Usage:
        model = MyModel()
        model = DistributedWrapper(model, device="cuda")
        # Inside training loop:
        loss.backward()
        model.sync_gradients()
        optimizer.step()
    """

    def __init__(
        self,
        module: Module,
        device: str = "cpu",
        sync_bn: bool = False,
    ):
        super().__init__()
        self.module = module
        self._modules["module"] = module
        self._device = device
        self._sync_bn = sync_bn
        self.world_size = get_world_size()
        self.rank = get_rank()

        if device != "cpu":
            self.module.to(device)

        if self.world_size > 1 and self.rank == 0:
            print(
                f"[DistributedWrapper] rank={self.rank} world={self.world_size} "
                f"device={device} sync_bn={sync_bn}"
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self):
        return self.module.named_parameters()

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.module.load_state_dict(state_dict)

    def to(self, device: str):
        self._device = device
        self.module.to(device)
        return self

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def sync_gradients(self):
        """All-reduce gradients across ranks and divide by world_size."""
        if self.world_size <= 1:
            return
        for p in self.module.parameters():
            if p.grad is not None:
                all_reduce(p.grad, "sum")
                p.grad.data = p.grad.data / self.world_size

    def barrier(self):
        if self.world_size > 1:
            barrier()

    @property
    def device(self) -> str:
        return self._device


__all__ = ["DistributedWrapper"]
