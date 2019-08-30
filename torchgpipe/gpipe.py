"""The GPipe implementation."""
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor
import torch.autograd
import torch.cuda
import torch.nn as nn

from torchgpipe.batchnorm import DeferredBatchNorm
from torchgpipe.microbatch import check, gather, scatter
from torchgpipe.pipeline import pipeline
from torchgpipe.stream import AbstractStream, new_stream

__all__ = ['GPipe']


Device = Union[torch.device, int, str]
Devices = Union[Iterable[Device], List[Device]]

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

if TYPE_CHECKING:
    Module = nn.Module[TensorOrTensors]
    NamedModules = OrderedDict[str, Module]
else:
    Module = nn.Module
    NamedModules = OrderedDict


def recommend_torchgpipe_balancing(title: str) -> ValueError:
    """Creates a :exc:`ValueError` with recommendation to
    :mod:`torchgpipe_balancing`.
    """
    return ValueError('''{title}

If your model is still under development, its optimal balance would change
frequently. In this case, we highly recommend torchgpipe_balancing for naive
automatic balancing:

  from torchgpipe import GPipe
  from torchgpipe_balancing import balance_by_time

  sample = torch.rand(...)
  balance = balance_by_time(model, sample, partitions=...)

  model = GPipe(model, balance, chunks=...)
'''.format(title=title))


MOVING_DENIED = TypeError('denied to move parameters and buffers, '
                          'because GPipe should manage device placement')


class GPipe(Module):
    """Wraps an arbitrary :class:`~torch.nn.Sequential` module to train on
    GPipe_. If the module requires lots of memory, GPipe will be very
    efficient::

        model = nn.Sequential(a, b, c, d)
        model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)
        output = model(input)

    .. _GPipe: https://arxiv.org/abs/1811.06965

    GPipe combines pipeline parallelism with checkpointing to reduce peak
    memory required to train while minimizing device under-utilization.

    You should determine the balance when defining a GPipe module, as balancing
    will not be done automatically. The module will be partitioned into
    multiple devices according to the given balance. You may rely on heuristics
    to find your own optimal configuration.

    Args:
        module (nn.Sequential):
            sequential module to be parallelized
        balance (ints):
            list of number of layers in each partition

    Keyword Args:
        devices (iterable of devices):
            devices to use (default: all CUDA devices)
        chunks (int):
            number of micro-batches (default: ``1``)
        checkpoint (str):
            when to enable checkpointing, one of ``'always'``,
            ``'except_last'``, or ``'never'`` (default: ``'except_last'``)
        deferred_batch_norm (bool):
            whether to use deferred BatchNorm moving statistics
            (default: ``False``, See :ref:`Deferred BatchNorm` for more details)

    Raises:
        TypeError:
            the module is not a :class:`~torch.nn.Sequential`.
        ValueError:
            invalid arguments, or wrong balance
        IndexError:
            the number of devices is fewer than the number of partitions.

    """

    #: The devices mapped to each partition.
    #:
    #: ``devices[-1]`` refers to the device of the last partition, which means
    #: it is the output device. Probably, you need to use it to transfer the
    #: target to calculate the loss without a device mismatch
    #: :exc:`RuntimeError`. For example::
    #:
    #:     out_device = gpipe.devices[-1]
    #:
    #:     for input, target in loader:
    #:         target = target.to(out_device, non_blocking=True)
    #:         output = gpipe(input)
    #:         loss = F.cross_entropy(output, target)
    #:
    devices: Tuple[torch.device, ...] = ()
    #                                 ^^^^
    # The default value () required for Sphinx's autoattribute.

    def __init__(self,
                 module: nn.Sequential,
                 balance: Optional[Iterable[int]] = None,
                 *,
                 devices: Optional[Devices] = None,
                 chunks: int = 1,
                 checkpoint: str = 'except_last',
                 deferred_batch_norm: bool = False,
                 ) -> None:
        super().__init__()

        if not isinstance(module, nn.Sequential):
            raise TypeError('non-sequential module cannot be partitioned')

        if balance is None:
            raise recommend_torchgpipe_balancing('balance is required')

        if chunks <= 0:
            raise ValueError('number of chunks must be positive integer')

        if checkpoint not in ['always', 'except_last', 'never']:
            raise ValueError("checkpoint is not one of 'always', 'except_last', or 'never'")

        self.chunks = chunks
        self.checkpoint = checkpoint

        if deferred_batch_norm:
            module = DeferredBatchNorm.convert_deferred_batch_norm(module, self.chunks)

        # Split the module into multiple partitions.
        balance = list(balance)

        if devices is None:
            devices = range(torch.cuda.device_count())
        devices = [torch.device(d) for d in devices]

        try:
            self.partitions, self.balance, self.devices = \
                self._split_module(module, balance, devices)
        except ValueError as exc:
            raise recommend_torchgpipe_balancing(str(exc))

        self._copy_streams: Tuple[Tuple[AbstractStream, ...], ...] = ()

    @staticmethod
    def _split_module(module: nn.Sequential,
                      balance: List[int],
                      devices: List[torch.device],
                      ) -> Tuple[List[nn.Sequential], Tuple[int, ...], Tuple[torch.device, ...]]:
        """Splits a module into multiple partitions.

        Returns:
            A tuple of (partitions, balance, devices).

            Partitions are represented as a :class:`~torch.nn.ModuleList` whose
            item is a partition. All layers in a partition are placed in the
            same device.

        Raises:
            ValueError:
                wrong balance
            IndexError:
                the number of devices is fewer than the number of partitions.

        """
        if len(module) != sum(balance):
            raise ValueError('module and sum of balance have different length '
                             '(module: %d, sum of balance: %d)' % (len(module), sum(balance)))
        if any(x <= 0 for x in balance):
            raise ValueError('all balance numbers must be positive integer '
                             '(balance: %r)' % balance)

        if len(balance) > len(devices):
            raise IndexError('too few devices to hold given partitions '
                             '(devices: %s, partitions: %d)' % (len(devices), len(balance)))

        i = 0
        partitions = []
        layers: NamedModules = OrderedDict()

        for name, layer in module.named_children():
            layers[name] = layer

            if len(layers) == balance[i]:
                # Group buffered layers as a partition.
                partition = nn.Sequential(layers)

                device = devices[i]
                partition.to(device)

                partitions.append(partition)

                # Prepare for the next partition.
                layers.clear()
                i += 1

        partitions = cast(List[nn.Sequential], nn.ModuleList(partitions))
        del devices[i:]

        return partitions, tuple(balance), tuple(devices)

    def __len__(self) -> int:
        """Counts the length of the underlying sequential module."""
        return sum(len(p) for p in self.partitions)

    def __getitem__(self, index: int) -> nn.Module:
        """Gets a layer in the underlying sequential module."""
        partitions = self.partitions
        if index < 0:
            partitions = partitions[::-1]

        for partition in partitions:
            try:
                return partition[index]
            except IndexError:
                pass

            shift = len(partition)

            if index < 0:
                index += shift
            else:
                index -= shift

        raise IndexError

    # GPipe should manage the device of each partition.
    # Deny cuda(), cpu(), and to() with device, by TypeError.
    def cuda(self, device: Optional[Device] = None) -> 'GPipe':
        raise MOVING_DENIED

    def cpu(self) -> 'GPipe':
        raise MOVING_DENIED

    def to(self, *args: Any, **kwargs: Any) -> 'GPipe':
        # Deny these usages:
        #
        # - to(device[, dtype, non_blocking])
        # - to(tensor[, non_blocking])
        #
        # But allow this:
        #
        # - to(dtype[, non_blocking])
        #
        if 'device' in kwargs or 'tensor' in kwargs:
            raise MOVING_DENIED

        if args:
            if isinstance(args[0], (torch.device, int, str)):
                raise MOVING_DENIED
            if torch.is_tensor(args[0]):
                raise MOVING_DENIED

        return super().to(*args, **kwargs)

    def _ensure_copy_streams(self) -> Tuple[Tuple[AbstractStream, ...], ...]:
        if not self._copy_streams:
            copy_streams = []

            for device in self.devices:
                copy_streams.append(tuple([new_stream(device) for _ in range(self.chunks)]))

            self._copy_streams = tuple(copy_streams)

        return self._copy_streams

    def forward(self, input: TensorOrTensors) -> TensorOrTensors:  # type: ignore
        """:class:`GPipe` is a fairly transparent module wrapper. It doesn't
        modify the input and output signature of the underlying module. But
        there's type restriction. Input and output have to be a
        :class:`~torch.Tensor` or a tuple of tensors. This restriction is
        applied at partition boundaries too.

        Args:
            input (tensor or tensors): input mini-batch

        Returns:
            tensor or tensors: output mini-batch

        Raises:
            TypeError: input is not a tensor or tensors.

        """
        check(input)

        if not self.devices:
            # Empty sequential module is not illegal.
            return input

        # Prepare separate CUDA streams only for copy.
        copy_streams = self._ensure_copy_streams()

        # Divide a mini-batch into micro-batches.
        batches = scatter(input, self.chunks)

        # The micro-batch index where the checkpointing stops.
        if self.training:
            checkpoint_stop = {'always': self.chunks,
                               'except_last': self.chunks-1,
                               'never': 0}[self.checkpoint]
        else:
            checkpoint_stop = 0

        # Run pipeline parallelism.
        pipeline(batches,
                 self.partitions,
                 self.devices,
                 copy_streams,
                 checkpoint_stop)

        # Merge the micro-batches into one mini-batch.
        output = gather(batches)
        return output
