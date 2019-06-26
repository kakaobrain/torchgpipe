"""The GPipe implemtation."""
from collections import OrderedDict
from queue import PriorityQueue
import sys
import threading
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union, cast

import torch
from torch import Tensor
import torch.autograd
import torch.cuda
import torch.nn as nn

from torchgpipe.batchnorm import DeferredBatchNorm
from torchgpipe.checkpoint import first
from torchgpipe.microbatch import check, gather, scatter
from torchgpipe.partition import Partition

__all__ = ['GPipe', 'current_microbatch']


Device = Union[torch.device, int, str]
Devices = Union[Iterable[Device], List[Device]]

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]
ChunkedTensorOrTensors = Union[List[Tensor], List[Tensors]]


class Message(NamedTuple):
    """Workers communicate with this message."""
    i: int
    payload: Any

    def __lt__(self, other: tuple) -> bool:
        # A message is enqueued in a priority queue. Only ``i`` is used to
        # determine a message priority. Therefore, we can store any object
        # including uncomparable user object into ``payload``. That's why we
        # need this class instead of a built-in tuple.
        other_i, _ = other
        return self.i < other_i


# The micro-batch leaf tensor storage for each partition worker thread.
_local = threading.local()


def current_microbatch() -> Optional[Tensor]:
    """Gets the current micro-batch identifier as a tensor.

    If your module relies on where the current micro-batch lane, use it to
    identify the lane.

    Returns:
        tensor or ``None``: A tensor which identifies the current micro-batch
        lane, or ``None`` for out of a GPipe context.

    .. seealso:: :ref:`Long Skip Connections`

    """
    try:
        return _local.microbatch
    except AttributeError:
        return None


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


class GPipe(nn.Module):
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
            self.partitions, self.balance, self.devices = self._partition(module, balance, devices)
        except ValueError as exc:
            raise recommend_torchgpipe_balancing(str(exc))

    def __len__(self) -> int:
        """Counts the length of the underlying sequential module."""
        partitions = cast(List[Partition], self.partitions)
        return sum(len(p) for p in partitions)

    def __getitem__(self, index: int) -> nn.Module:
        """Gets a layer in the underlying sequential module."""
        partitions = cast(List[Partition], self.partitions)
        if index < 0:
            partitions = cast(List[Partition], reversed(partitions))

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
    def cuda(self, device: Any = None) -> nn.Module:
        raise MOVING_DENIED

    def cpu(self) -> nn.Module:
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
            if isinstance(args[0], Tensor):
                raise MOVING_DENIED

        return super().to(*args, **kwargs)

    @staticmethod
    def _partition(module: nn.Sequential,
                   balance: List[int],
                   devices: List[torch.device],
                   ) -> Tuple[nn.ModuleList, Tuple[int, ...], Tuple[torch.device, ...]]:
        """Partitions the given sequential module onto the devices.

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
        layers_buffer: Dict[str, nn.Module] = OrderedDict()

        for name, layer in module.named_children():
            layers_buffer[name] = layer

            if len(layers_buffer) == balance[i]:
                # Group buffered layers as a partition.
                partition_module = nn.Sequential(layers_buffer)

                device = devices[i]
                partition = Partition(partition_module, device)
                partition.to(device)

                partitions.append(partition)

                # Prepare for the next partition.
                layers_buffer.clear()
                i += 1

        del devices[i:]

        return nn.ModuleList(partitions), tuple(balance), tuple(devices)

    def _spawn_workers(self) -> Tuple[PriorityQueue, PriorityQueue]:
        """Creates worker threads."""
        partitions = cast(List[Partition], self.partitions)

        n = len(partitions)
        queues: List[PriorityQueue] = [PriorityQueue() for _ in range(n+1)]
        grad_enabled = torch.is_grad_enabled()

        for i, partition in enumerate(partitions):
            in_queue = queues[i]
            out_queue = queues[i+1]

            args = (partition, in_queue, out_queue, grad_enabled)
            t = threading.Thread(target=GPipe._worker, args=args)
            t.daemon = True
            t.start()

        return queues[0], queues[-1]

    @staticmethod
    def _worker(partition: Partition,
                in_queue: PriorityQueue,
                out_queue: PriorityQueue,
                grad_enabled: bool,
                ) -> None:
        """Run by worker threads."""
        torch.set_grad_enabled(grad_enabled)

        input: TensorOrTensors
        checkpoint: bool
        recompute: Optional[Tensor] = None
        failed = False

        while True:
            # Block until getting a micro-batch.
            msg = in_queue.get()

            # None means end-of-mini-batch.
            if msg.payload is None:
                out_queue.put(msg)
                break

            # The previous partition sent an exception (msg = exc_info.)
            if msg.i == -1:
                out_queue.put(msg)
                continue

            # Ignore every input after an error.
            if failed:
                continue

            input, leaf, checkpoint = msg.payload

            # Track the current micro-batch lane by the leaf tensor of the
            # lane. It can be accessed by current_microbatch().
            _local.microbatch = leaf

            # Linearize micro-batches by dependency between nth micro-batch
            # input and n-1th micro-batch output. It prevents unexpected
            # recursive backward from checkpoints.
            input = first(input, recompute)

            try:
                output, recompute = partition(input, checkpoint)
            except BaseException:
                # A user-exception occurred. Pass it to the next partition to
                # make the main thread detect it as soon as possible.
                exc_info = sys.exc_info()
                error = Message(-1, exc_info)
                out_queue.put(error)
                del error

                # The main thread will close this worker soon.
                failed = True
                continue
            finally:
                # Let the former partition detect that
                # it's ready to receive a new micro-batch or message.
                in_queue.task_done()

            # Micro-batch lockstep: During this partition is executing a micro-batch, to copy
            # a micro-batch by the next partition would be blocked. To prevent the blocking,
            # don't send the current micro-batch until the next partition is ready to receive it.
            out_queue.join()

            msg = Message(msg.i, (output, leaf, checkpoint))
            out_queue.put(msg)

    def _push_input(self,
                    input: TensorOrTensors,
                    in_queue: PriorityQueue,
                    ) -> int:
        """Pushes chunked inputs to the first partition."""
        # Divide a mini-batch into micro-batches.
        in_device = self.devices[0]
        inputs = scatter(input, chunks=self.chunks, device=in_device)

        # The number of inputs might be smaller than the number of chunks.
        num_inputs = len(inputs)

        for i, _input in enumerate(inputs):
            # NOTE(sublee): 'except_last' is the defualt option. Compare it first.
            if self.checkpoint == 'except_last':
                checkpoint = (i < self.chunks-1)
            elif self.checkpoint == 'always':
                checkpoint = True
            elif self.checkpoint == 'never':
                checkpoint = False

            # Every partition should track the current micro-batch. A
            # micro-batch lane can be identified its detached leaf tensor.
            leaf = (_input[0] if isinstance(_input, tuple) else _input).detach()

            msg = Message(i, (_input, leaf, checkpoint))
            in_queue.put(msg)

        close = Message(num_inputs, None)
        in_queue.put(close)

        return num_inputs

    def _pull_output(self,
                     num_inputs: int,
                     in_queue: PriorityQueue,
                     out_queue: PriorityQueue,
                     ) -> Tensor:
        """Collects and concatenates chunked outputs from the last partition.

        If an exception from a parititon is detected, all workers are closed
        and the exception is re-raised.

        Raises:
            Exception: any exception from a partition

        """
        outputs = []
        for _ in range(num_inputs):
            msg = out_queue.get()
            out_queue.task_done()

            if msg.i == -1:
                # Close worker threads immediately.
                close = Message(-1, None)
                in_queue.put(close)
                out_queue.get()

                # Raise the exception from a partition.
                exc_info = msg.payload
                raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

            output, _, _ = msg.payload
            outputs.append(output)

        out_device = self.devices[-1]
        output = gather(outputs, device=out_device)
        out_queue.get()

        return output

    def forward(self, input: TensorOrTensors) -> TensorOrTensors:
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
        if not self.devices:
            # An empty sequential module is wrapped. Empty sequential module is
            # not illegal. Just check the input type.
            check(input)
            return input

        in_queue, out_queue = self._spawn_workers()
        num_inputs = self._push_input(input, in_queue)
        return self._pull_output(num_inputs, in_queue, out_queue)
