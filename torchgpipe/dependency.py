"""Autograd functions for making dependency between two tensors."""
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from torchgpipe.stream import default_stream, use_stream

__all__: List[str] = []


class Phonies:
    """Manages phony tensors."""

    def __init__(self) -> None:
        self.phonies: Dict[torch.device, Tensor] = {}

    def __getitem__(self, device: torch.device) -> Tensor:
        try:
            return self.phonies[device]
        except KeyError:
            with use_stream(default_stream(device)):
                phony = torch.empty(0, device=device, requires_grad=True)
            self.phonies[device] = phony
            return phony


# The common interface between :class:`Fork` and :class:`Join`.
class Context:
    pass


class Fork(torch.autograd.Function):
    # TODO(sublee, frost): Explain without term "lane".
    """Makes a branch at the given lane by a phony tensor. A phony tensor is a
    tensor of size 0, separate from the input. In contrast to the input, its
    implicit gradient accumulation is very cheap and safe to any custom
    streams.

    .. sourcecode:: text

        input -- Fork -- input
                   |
                   +---- phony

    """
    phonies = Phonies()

    @staticmethod
    def forward(ctx: Context, input: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore
        phony = Fork.phonies[input.device]
        return input.detach(), phony.detach()

    @staticmethod
    def backward(ctx: Context, grad_input: Tensor, grad_phony: Tensor) -> Tensor:  # type: ignore
        return grad_input


class Join(torch.autograd.Function):
    # TODO(sublee, frost): Explain without term "lane".
    """Makes the given lane depend on the branch which is represented as a
    phony tensor by :class:`Fork`.

    .. sourcecode:: text

        input1 -- Fork --------------- input1
                    |
                    +-- phony --+
                                |
        input2 --------------- Join -- input2

    """
    @staticmethod
    def forward(ctx: Context, input: Tensor, phony: Tensor) -> Tensor:  # type: ignore
        return input.detach()

    @staticmethod
    def backward(ctx: Context, grad_input: Tensor) -> Tuple[Tensor, None]:  # type: ignore
        return grad_input, None
