from typing import TYPE_CHECKING, Iterator, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from torchgpipe.checkpoint import checkpoint as do_checkpoint

__all__ = ['Partition']

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

if TYPE_CHECKING:
    Module = nn.Module[Tuple[TensorOrTensors, Optional[Tensor]]]
else:
    Module = nn.Module


class Partition(nn.Module):
    """Wraps a module and occupies one device.

    It has a responsibility at the partition boundary. It will copy the input
    into the occupied device and make a checkpoint to save peak memory usage.

    Args:
        module (nn.Module): underlying module
        device (torch.device): device where this partition occupies

    """

    def __init__(self, module: nn.Sequential, device: torch.device) -> None:
        super().__init__()

        self.module = module
        self.device = device

    def __iter__(self) -> Iterator[nn.Module]:
        yield from self.module

    def __len__(self) -> int:
        return len(self.module)

    def __getitem__(self, index: int) -> nn.Module:
        return self.module[index]

    def bring(self, input: TensorOrTensors) -> TensorOrTensors:
        """Brings an input to this partition.

        1. Transfer the input to the device where the partition placed.
        2. Turn on ``requires_grad`` of the input to use checkpointing.

        """
        if isinstance(input, tuple):
            return tuple(x.to(self.device).requires_grad_() for x in input)
        return input.to(self.device).requires_grad_()

    def forward(self,  # type: ignore
                input: TensorOrTensors,
                checkpoint: bool = True,
                ) -> Tuple[TensorOrTensors, Optional[Tensor]]:
        input = self.bring(input)

        # If checkpoint=False, don't make a checkpoint. This conditional
        # behavior is useful to reduce recomputation overhead at the last
        # micro-batches.
        if not checkpoint or not self.training:
            return self.module(input), None

        output, recompute = do_checkpoint(self.module, input)
        return output, recompute
