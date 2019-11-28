"""Converts skip connections with ``@skippable`` to ``tuple``."""
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Deque, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor, nn
from torchgpipe import is_checkpointing, is_recomputing
from torchgpipe.skip import Namespace
from torchgpipe.skip.skippable import Skippable

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

if TYPE_CHECKING:
    Seats = OrderedDict[Tuple[Namespace, str], int]
else:
    Seats = OrderedDict


class SkipsAsTuple(nn.Module):
    """The base module for old-fashioned skip connections. It handles arguments
    including the input and the skips.
    """

    def __init__(self,
                 num_skips: int,
                 unpack_input: Deque[bool],
                 unpack_output: Deque[bool],
                 ) -> None:
        super().__init__()
        self.num_skips = num_skips
        self.unpack_input = unpack_input
        self.unpack_input_for_recomputing: List[bool] = []
        self.unpack_output = unpack_output

    def forward(self, input_skips: TensorOrTensors) -> TensorOrTensors:  # type: ignore
        input: TensorOrTensors = input_skips
        skips: Tensors = ()

        # num_skips=0 means that this module is the first module of the skip
        # connections.
        if self.num_skips:
            assert isinstance(input_skips, tuple)

            input = input_skips[:-self.num_skips]
            skips = input_skips[-self.num_skips:]

            if is_recomputing():
                unpack_input = self.unpack_input_for_recomputing.pop()
            else:
                unpack_input = self.unpack_input.popleft()
                if is_checkpointing():
                    self.unpack_input_for_recomputing.append(unpack_input)

            if unpack_input:
                input = input[0]

        output, skips = self._forward(input, skips)

        unpack_output = torch.is_tensor(output)
        self.unpack_output.append(unpack_output)

        if not skips:
            return output

        if unpack_output:
            return (cast(Tensor, output), *skips)
        else:
            return output + skips

    def _forward(self, input: TensorOrTensors, skips: Tensors) -> Tuple[TensorOrTensors, Tensors]:
        # Interface: (input, skips) -> (output, skips)
        raise NotImplementedError


class Gutter(SkipsAsTuple):
    """Just passes the incoming skips."""

    def __init__(self,
                 module: nn.Module,
                 num_skips: int,
                 unpack_input: Deque[bool],
                 unpack_output: Deque[bool],
                 ) -> None:
        super().__init__(num_skips, unpack_input, unpack_output)
        self.module = module

    def _forward(self, input: TensorOrTensors, skips: Tensors) -> Tuple[TensorOrTensors, Tensors]:
        output = self.module(input)
        return output, skips


class Branch(SkipsAsTuple):
    """Stashes or pops skips."""

    def __init__(self,
                 skippable: Skippable,
                 prev_seats: Seats,
                 next_seats: Seats,
                 num_skips: int,
                 unpack_input: Deque[bool],
                 unpack_output: Deque[bool],
                 ) -> None:
        super().__init__(num_skips, unpack_input, unpack_output)
        self.skippable = skippable
        self.prev_seats = prev_seats
        self.next_seats = next_seats

    def _forward(self, input: TensorOrTensors, skips: Tensors) -> Tuple[TensorOrTensors, Tensors]:
        stashed = {}

        def handle_stash(name: str, tensor: Optional[Tensor]) -> None:
            ns, name = self.skippable.namespaced(name)
            stashed[(ns, name)] = tensor

        def handle_pop(name: str) -> Optional[Tensor]:
            ns, name = self.skippable.namespaced(name)
            i = self.prev_seats[(ns, name)]
            return skips[i]

        output = self.skippable.dispatch(input, handle_stash, handle_pop)

        next_skips = []
        for ns, name in self.next_seats:
            if (ns, name) in stashed:
                skip = stashed[(ns, name)]
            else:
                i = self.prev_seats[(ns, name)]
                skip = skips[i]
            assert skip is not None
            next_skips.append(skip)

        return output, tuple(next_skips)


def tuplify_skips(module: nn.Sequential) -> nn.Sequential:
    """Converts ``@skippable`` modules and intermediate modules between
    ``@skippable``s to represent the skips as ``tuple``.
    """
    seats: Seats = OrderedDict()
    next_seats = seats
    unpack_output: Deque[bool] = deque()
    layers: List[nn.Module] = []

    for layer in module.children():
        num_skips = len(seats)
        unpack_input = unpack_output
        unpack_output = deque()

        if isinstance(layer, Skippable):
            for ns, name in layer.stashable():
                seats[(ns, name)] = -1
            for ns, name in layer.poppable():
                del seats[(ns, name)]
            for i, (ns, name) in enumerate(seats):
                seats[(ns, name)] = i

            prev_seats, next_seats = next_seats, seats.copy()

            layer = Branch(layer, prev_seats, next_seats, num_skips, unpack_input, unpack_output)

        elif seats:
            layer = Gutter(layer, num_skips, unpack_input, unpack_output)

        layers.append(layer)
    return nn.Sequential(*layers)
