"""Checkpointing with preceding recomputaiton."""
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.autograd

__all__ = ['checkpoint', 'first']


Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]
Function = Callable[[TensorOrTensors], TensorOrTensors]


def checkpoint(module: Function, input: TensorOrTensors) -> Tuple[TensorOrTensors, Tensor]:
    """Makes a checkpoint at the given module.

    It is very similar to :func:`torch.utils.checkpoint.checkpoint` but it
    provides a recomputation autograd edge together to allow recomputing before
    the checkpoint.

    The recomputation edge is represented as a dummy output. If you make the
    output depend on the recomputation dummy output, the recomputation can
    precede the checkpoint::

        output1, recompute = checkpoint(model, input1)

        input2 = first(input2, recompute)
        output2 = model(input2)

        output = torch.cat((output1, output2))

    Returns:
        A tuple of (output, recompute).

    """
    result = Result()

    if isinstance(input, tuple):
        unwrap_input = False
    else:
        input = (input,)
        unwrap_input = True

    output = Checkpoint.apply(result, module, unwrap_input, *input)
    dummy = output[0] if isinstance(output, tuple) else output
    recompute = Recompute.apply(dummy, result, module, unwrap_input, *input)

    return output, recompute


class Result:
    """A shared memory between :class:`Checkpoint` and :class:`Recompute`."""
    __slots__ = ('value',)

    def __init__(self) -> None:
        self.value: Any = None

    def set(self, value: Any) -> None:  # pragma: no cover
        self.value = value

    def get(self) -> Any:  # pragma: no cover
        return self.value


class Context:
    """A common interface between the :class:`Checkpoint` and
    :class:`Recompute` context.
    """
    result: Result

    # NOTE(sublee): 'module' cannot be annotated with 'Function' because mypy
    # infers this attribute as an instance method. That's why this is annotated
    # with 'Any' instead.
    # See: https://github.com/python/mypy/issues/708.
    module: Any

    unwrap_input: bool

    saved_tensors: Tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None:  # pragma: no cover
        pass


class Checkpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Context,
                result: Result,
                module: Function,
                unwrap_input: bool,
                *input: Tensor,
                ) -> TensorOrTensors:
        ctx.result = result

        ctx.module = module
        ctx.unwrap_input = unwrap_input
        ctx.save_for_backward(*input)

        with torch.no_grad():
            output = module(input[0] if unwrap_input else input)

        return output

    @staticmethod
    def backward(ctx: Context,
                 *grad_output: Tensor,
                 ) -> Tuple[Optional[Tensor], ...]:  # pragma: no cover
        output, input_leaf = recompute_once(ctx)

        if isinstance(output, tuple):
            torch.autograd.backward(output, grad_output)
        else:
            output.backward(grad_output[0])

        grad_input: List[Optional[Tensor]] = [None, None, None]
        grad_input.extend(x.grad for x in input_leaf)
        return tuple(grad_input)


class Recompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Context,
                dummy: Tensor,
                result: Result,
                module: Function,
                unwrap_input: bool,
                *input: Tensor,
                ) -> Tensor:
        ctx.result = result

        ctx.module = module
        ctx.unwrap_input = unwrap_input
        ctx.save_for_backward(*input)

        return dummy

    @staticmethod
    def backward(ctx: Context, *grad_output: Tensor) -> Tuple[None, ...]:  # pragma: no cover
        _ = grad_output
        recompute_once(ctx)
        return (None,) * (len(ctx.saved_tensors) + 4)


def recompute_once(ctx: Context) -> Tuple[TensorOrTensors, Tensors]:  # pragma: no cover
    """Ensures the recomputation only once."""
    already_recomputed = ctx.result.get()
    if already_recomputed:
        return already_recomputed

    input = ctx.saved_tensors
    input_leaf = tuple(x.detach().requires_grad_() for x in input)

    with torch.enable_grad():
        output = ctx.module(input_leaf[0] if ctx.unwrap_input else input_leaf)

    ctx.result.set((output, input_leaf))

    return output, input_leaf


def first(input: TensorOrTensors, dummy: Optional[Tensor]) -> TensorOrTensors:
    """Makes the input depend on the dummy. It injects a phony dependency
    between them.
    """
    if dummy is None:
        # Just pass.
        return input

    if isinstance(input, tuple):
        head = First.apply(input[0], dummy)
        return (head,) + input[1:]

    return First.apply(input, dummy)


class First(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Context, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        _ = tensor2
        return tensor1

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:  # pragma: no cover
        return grad_output, None
