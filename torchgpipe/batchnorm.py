"""Tracks the moving average for each mini-batch instead of micro-batch."""
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm as BatchNorm

from torchgpipe.checkpoint import is_recomputing

__all__ = ['patch_deferred_batch_norm']


def patch_deferred_batch_norm(module: nn.Module) -> None:
    """Apply this function to a module to patch the BatchNorm layers to become deferred::

        model = resnet101()
        model.apply(patch_deferred_batch_norm)

    """
    if isinstance(module, BatchNorm):
        hook = DeferredBatchNormHook()
        hook(module)


class DeferredBatchNormHook:
    def __init__(self) -> None:
        self.tracked = False

    def __call__(self, bn: BatchNorm) -> None:
        if not bn.track_running_stats or bn.momentum is None:
            # The given batch norm doesn't track running stats.
            return

        bn.register_buffer('sum', torch.zeros_like(bn.running_mean))
        bn.register_buffer('sum_squares', torch.zeros_like(bn.running_var))
        bn.register_buffer('counter', torch.tensor(0, dtype=torch.long))

        bn.register_forward_pre_hook(self.forward_pre_hook)
        bn.register_forward_hook(self.forward_hook)
        bn.register_backward_hook(self.backward_hook)

    def forward_pre_hook(self, bn: BatchNorm, inputs: Tuple[Tensor, ...]) -> None:
        if not (bn.training and bn.track_running_stats):
            return

        # Don't track the running stats of this batch. It is already deferred.
        bn.track_running_stats = False
        bn.momentum_ = bn.momentum
        bn.momentum = None

        # Skip if this forward pass is triggered by checkpoint recomputation.
        if is_recomputing():
            return

        input, = inputs

        # Detach from the autograd graph.
        input = input.detach()

        # Dimensions except channel. For example, (0, 2, 3) is for BatchNorm2d.
        dim = [0]
        dim.extend(range(2, input.dim()))

        bn.sum += input.sum(dim)
        bn.sum_squares += (input**2).sum(dim)

        size = input.size().numel() / input.size(1)
        bn.counter += size

        # Enable the backward hook.
        self.tracked = True

    def forward_hook(self, bn: BatchNorm, input: Tensor, output: Tensor) -> None:
        # Any internal state modified by this hook should not be visible to users.
        bn.track_running_stats = True
        try:
            bn.momentum = bn.momentum_
        except AttributeError:
            pass
        else:
            del bn.momentum_

    def backward_hook(self, bn: BatchNorm,
                      grad_input: Tensor,
                      grad_output: Tensor) -> None:  # pragma: no cover
        if not self.tracked:
            return

        new_mean = bn.sum/bn.counter
        new_var = bn.sum_squares/bn.counter - new_mean**2

        # Calculate the exponential moving average here.
        bn.running_mean = bn.running_mean*(1-bn.momentum) + new_mean*bn.momentum
        bn.running_var = bn.running_var*(1-bn.momentum) + new_var*bn.momentum

        bn.sum.zero_()
        bn.sum_squares.zero_()
        bn.counter.zero_()

        # Disable the backward hook until the next forward pass.
        self.tracked = False
