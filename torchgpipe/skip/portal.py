"""A portal keeps a tensor in the pocket plane. The tensor becomes hidden to
the autograd engine. The link between two functions out of autograd is one of
the most important parts of skip connections.

The metaphor is inspired by Portal™ from Valve.

"""
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from torchgpipe.copy import Context as CopyContext
from torchgpipe.copy import Copy
from torchgpipe.phony import get_phony
from torchgpipe.stream import AbstractStream, get_device

__all__: List[str] = []


class Portal:
    """A portal for a tensor."""

    def __init__(self,
                 tensor: Optional[Tensor],
                 grad: Optional[Tensor] = None,
                 tensor_life: int = 1,
                 ) -> None:
        self.tensor = tensor
        self.grad = grad

        # How many times the tensor can be retrieved by use_tensor().
        # Typically, it is 1 because every stashed tensor must be popped only
        # once. But it should be 2 if checkpointing is enabled for
        # recomputation.
        self.tensor_life = tensor_life

    def blue(self) -> Tensor:
        """Creates a :class:`PortalBlue` which hides the underlying tensor from
        the autograd engine.

        The main lane of an autograd graph should join to
        the phony it returns::

            PortalBlue --+
                         |
            ---------- Join --

        """
        if self.tensor is None:
            raise RuntimeError('tensor in portal has been removed')

        return PortalBlue.apply(self, self.tensor)

    def orange(self, phony: Tensor) -> Tensor:
        """Creates a :class:`PortalOrange` which retrieves the hidden tensor
        without losing ability of backpropagation.

        Give a phony forked from the main lane of an autograd graph::

                +-- PortalOrange --+
                |                  |
            -- Fork --------- f(a, b) --

        """
        if self.tensor is None:
            raise RuntimeError('tensor in portal has been removed')

        return PortalOrange.apply(self, phony)

    def copy(self,
             prev_stream: AbstractStream,
             next_stream: AbstractStream,
             phony: Tensor,
             ) -> Tensor:
        """Copies the hidden tensor by a :class:`PortalCopy`.

        Give a phony and use the returning phony to keep backpropagation::

                +-- PortalCopy --+
                |                |
            -- Fork ---------- Join --

        """
        if self.tensor is None:
            raise RuntimeError('tensor in portal has been removed')

        return PortalCopy.apply(self, prev_stream, next_stream, phony)

    def keep_tensor(self, tensor: Optional[Tensor]) -> None:
        """Stores a tensor into this portal."""
        self.tensor = tensor

    def use_tensor(self) -> Tensor:
        """Retrieves the underlying tensor and decreases the tensor  life. When
        the life becomes 0, it the tensor will be removed.
        """
        if self.tensor is None:
            raise RuntimeError('tensor in portal has been removed')

        tensor = self.tensor

        self.tensor_life -= 1
        if self.tensor_life == 0:
            self.tensor = None

        return tensor

    def keep_grad(self, grad: Tensor) -> None:
        """Stores a gradient into this portal."""
        self.grad = grad

    def use_grad(self) -> Tensor:
        """Retrieves and removes the underlying gradient. The gradient is
        always ephemeral.
        """
        if self.grad is None:
            raise RuntimeError('grad in portal has been removed or never set')

        grad = self.grad
        self.grad = None
        return grad


# Common interface between :class:`PortalBlue`, :class:`PortalOrange`, and
# :class:`PortalCopy`.
class Context(CopyContext):
    portal: Portal


class PortalBlue(torch.autograd.Function):
    """Hides a tensor from the autograd engine by a :class:`Portal`."""
    @staticmethod
    def forward(ctx: Context,  # type: ignore
                portal: Portal,
                tensor: Tensor,
                ) -> Tensor:
        # This condition is guaranteed by Portal.blue(). However, this function
        # still has to take the tensor for passing its gradient.
        assert tensor is portal.tensor

        ctx.portal = portal

        phony = get_phony(tensor.device, requires_grad=False)
        return phony.detach()

    @staticmethod
    def backward(ctx: Context,  # type: ignore
                 grad_phony: Tensor,
                 ) -> Tuple[None, Tensor]:
        # The paired PortalOrange should keep the gradient.
        grad = ctx.portal.use_grad()
        return None, grad


class PortalOrange(torch.autograd.Function):
    """Retrieves the hidden tensor from a :class:`Portal`."""
    @staticmethod
    def forward(ctx: Context, portal: Portal, phony: Tensor) -> Tensor:  # type: ignore
        ctx.portal = portal
        tensor = portal.use_tensor()
        return tensor.detach()

    @staticmethod
    def backward(ctx: Context, grad: Tensor) -> Tuple[None, None]:  # type: ignore
        # The paired PortalBlue will use the gradient.
        ctx.portal.keep_grad(grad)
        return None, None


class PortalCopy(torch.autograd.Function):
    """Copies the hidden tensor in a :class:`Portal`. It replaces the hidden
    tensor with copied one.
    """
    @staticmethod
    def forward(ctx: Context,  # type: ignore
                portal: Portal,
                prev_stream: AbstractStream,
                next_stream: AbstractStream,
                phony: Tensor,
                ) -> Tensor:
        ctx.portal = portal

        if portal.tensor is not None:
            portal.tensor, = Copy.forward(ctx, prev_stream, next_stream, portal.tensor)

        phony = get_phony(get_device(next_stream), requires_grad=False)
        return phony.detach()

    @staticmethod
    def backward(ctx: Context,  # type: ignore
                 grad_phony: Tensor,
                 ) -> Tuple[None, None, None, None]:
        portal = ctx.portal

        if portal.grad is not None:
            _, _, portal.grad = Copy.backward(ctx, portal.grad)

        return None, None, None, None
