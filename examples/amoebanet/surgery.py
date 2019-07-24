"""Utility modules for breaking a complex module into sequential.

"""
from typing import List, Optional, Tuple

import torch
from torch import nn

__all__ = ['Concat', 'FirstAnd', 'InputOne', 'MergeTwo', 'Shift', 'Twin', 'TwinLast']


Tensors = Tuple[torch.Tensor, ...]


class Twin(nn.Module):
    """Duplicates the tensor::

         ┌──────┐
     a --│ Twin │--> a
         │   '--│--> a
         └──────┘

    """

    def forward(self, tensor: torch.Tensor) -> Tensors:
        return tensor, tensor


class TwinLast(nn.Module):
    """Duplicates the last tensor::

        a -----> a
        b -----> b
        c --+--> c
            +--> c'

    """

    def forward(self, tensors: Tensors) -> Tensors:
        return tensors + (tensors[-1],)


class InputOne(nn.Module):
    """Picks one tensor for the underlying module input::

        a -----> a
        b --f--> f(b)
        c -----> c

    """

    def __init__(self, module: nn.Module, i: int, insert: Optional[int] = None):
        super().__init__()
        self.module = module
        self.i = i
        self.insert = insert

    def forward(self, tensors: Tensors) -> Tensors:
        i = self.i

        input = tensors[i]
        output = self.module(input)

        if not isinstance(output, tuple):
            output = (output,)

        if self.insert is None:
            # Replace with the input.
            return tensors[:i] + output + tensors[i+1:]

        return tensors[:self.insert] + output + tensors[self.insert:]


class Shift(nn.Module):
    """Moves the last tensor ahead of the tensors::

            +--> c
        a --|--> a
        b --|--> b
        c --+

    """

    def forward(self, tensors: Tensors) -> Tensors:
        return (tensors[-1],) + tensors[:-1]


class MergeTwo(nn.Module):
    """Merges the last two tensors and replace them with the result::

        a -----> a
        b --+--> b+c
        c --+

    """

    def __init__(self, i: int, j: int):
        super().__init__()
        self.i = i
        self.j = j

    def forward(self, tensors: Tensors) -> Tensors:
        i = self.i
        j = self.j
        return tensors[:i] + (sum(tensors[i:j+1]),) + tensors[j+1:]


class FirstAnd(nn.Module):
    """Skips the first tensor, executes the underlying module by the remaining
    tensors::

        a -----> a
        b --+--> f(b, c)
        c --+

    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, tensors: Tensors) -> Tensors:
        output = self.module(tensors[1:])
        if not isinstance(output, tuple):
            output = (output,)
        return (tensors[0],) + output


class Concat(nn.Module):
    """Concat all tensors::

        a --+
        b --+--> concat(a, b, c)
        c --+

    """

    def __init__(self, indices: List):
        super().__init__()
        self.indices = indices

    def forward(self, tensors: Tensors) -> torch.Tensor:
        return torch.cat([tensors[i] for i in self.indices], dim=1)
