"""The GPipe implemtation."""
from queue import PriorityQueue
import sys
import threading
from typing import Any, Iterable, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.autograd
import torch.cuda
import torch.nn as nn

from torchgpipe.batchnorm import patch_deferred_batch_norm
from torchgpipe.checkpoint import first
from torchgpipe.microbatch import gather, scatter
from torchgpipe.partition import Partition

__all__ = ['GPipe']


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


class GPipe(nn.Module):
    """Implements `GPipe <https://arxiv.org/abs/1811.06965>`_ a scalable
    pipeline parallelism at the module level. GPipe aims to reduce peak memory
    usage while minimizing device under-untilization.

    This container separates a sequential module into several partitions over
    each device just like naive model parallelism. Its :meth:`forward` divides
    a mini-batch into multiple micro-batches. Each partition sends a
    micro-batch output to the next partition then takes the next micro-batch as
    soon as possible. So unlike naive model parallelism, those partitions can
    be overlapped with each other.

    GPipe also integrates checkpointing to futher reduce memory usage.
    Recomputation of a micro-batch can be evaluated preceding its backward
    pass.

    .. note::
        There are still idle time called "bubbling" so the utilization might be
        less than 100%. But the bubbling can be partly shrunk by increasing the
        number of micro-batches or balanced partitioning.

    Args:
        module (nn.Sequential):
            sequential module to be parallelized
        balance (ints):
            lengths of layers on each partition

    Keyword Args:
        devices (iterable of devices):
            devices to use (default: all CUDA devices)
        chunks (int):
            number of micro-batches (default: 1)
        checkpoint (str):
            when to enable checkpoints, one of 'always', 'except_last', or
            'never' (default: 'except_last')
        deferred_batch_norm (bool):
            enable deferred BatchNorm moving statistics (default: False)

    Example::

        >>> model = torch.nn.Sequential(l1, l2, l3, l4)
        >>> gpipe = GPipe(model, balance=[1, 1, 1, 1], chunks=8)
        >>> output = gpipe(input)

    """

    def __init__(self,
                 module: nn.Sequential,
                 balance: Iterable[int],
                 *,
                 devices: Optional[Devices] = None,
                 chunks: int = 1,
                 checkpoint: str = 'except_last',
                 deferred_batch_norm: bool = False):
        super().__init__()

        if deferred_batch_norm:
            module.apply(patch_deferred_batch_norm)

        self._partitions, self.balance, self.in_device, self.out_device = \
            self.partition(module, balance, devices)

        self.chunks = chunks

        if checkpoint not in ['always', 'except_last', 'never']:
            raise ValueError("checkpoint is not one of 'always', 'except_last', or 'never'")

        self.checkpoint = checkpoint

    def __iter__(self) -> Iterable[nn.Module]:
        """Iterates over underlying sequential layers."""
        for partition in self._partitions:
            yield from partition

    def __len__(self) -> int:
        """Counts the length of the underlying sequential module."""
        return sum(len(p) for p in self._partitions)

    def __getitem__(self, index: int) -> nn.Module:
        """Gets a layer in the underlying sequential module."""
        if index < 0:
            partitions = reversed(self._partitions)
        else:
            partitions = self._partitions

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

    def partitions(self) -> List[Partition]:
        """The underlying partitions."""
        return list(self._partitions)

    @staticmethod
    def partition(module: nn.Sequential,
                  balance: Iterable[int],
                  devices: Optional[Devices],
                  ) -> Tuple[nn.ModuleList, List[int], torch.device, torch.device]:
        """Partitions the given sequential module onto the devices.

        Returns:
            A tuple of (partitions, input device, output device).

            Partitions are represented as a :class:`~torch.nn.ModuleList` whose
            item is a partition. All layers in a partition are placed in the
            same device.

        """
        if not isinstance(module, nn.Sequential):
            raise TypeError('non-sequential module cannot be partitioned')

        balance = list(balance)

        if len(module) != sum(balance):
            raise ValueError('module and sum of balance have different length '
                             '(module: %d, sum of balance: %d)' % (len(module), sum(balance)))

        if devices is None:
            devices = [torch.device(d) for d in range(torch.cuda.device_count())]
        else:
            devices = [torch.device(d) for d in devices]

        if len(balance) > len(devices):
            raise ValueError('too few devices to hold given partitions '
                             '(devices: %s, paritions: %d)' % (len(devices), len(balance)))

        i = 0
        partitions = []
        partition_layers = []

        for layer in module:
            partition_layers.append(layer)

            if len(partition_layers) == balance[i]:
                # Group buffered layers as a partition.
                partition_module = nn.Sequential(*partition_layers)

                device = devices[i]
                partition = Partition(partition_module, device)
                partition.to(device)

                partitions.append(partition)

                # Prepare for the next partition.
                del partition_layers[:]
                i += 1

        in_device = partitions[0].device
        out_device = partitions[-1].device

        return nn.ModuleList(partitions), balance, in_device, out_device

    def spawn_workers(self) -> Tuple[PriorityQueue, PriorityQueue]:
        """Creates worker threads."""
        n = len(self._partitions)
        queues: List[PriorityQueue] = [PriorityQueue() for _ in range(n+1)]
        grad_enabled = torch.is_grad_enabled()

        for i, partition in enumerate(self._partitions):
            in_queue = queues[i]
            out_queue = queues[i+1]

            args = (partition, in_queue, out_queue, grad_enabled)
            t = threading.Thread(target=GPipe.worker, args=args)
            t.daemon = True
            t.start()

        return queues[0], queues[-1]

    @staticmethod
    def worker(partition: Partition,
               in_queue: PriorityQueue,
               out_queue: PriorityQueue,
               grad_enabled: bool,
               ) -> None:
        """A worker thread runs it."""
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

            input, checkpoint = msg.payload

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

            msg = Message(msg.i, (output, checkpoint))
            out_queue.put(msg)

    def push_input(self,
                   input: TensorOrTensors,
                   in_queue: PriorityQueue,
                   ) -> int:
        """Pushes chunked inputs to the first partition."""
        # Divide a mini-batch into micro-batches.
        inputs = scatter(input, chunks=self.chunks, device=self.in_device)

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

            msg = Message(i, (_input, checkpoint))
            in_queue.put(msg)

        close = Message(num_inputs, None)
        in_queue.put(close)

        return num_inputs

    def pull_output(self,
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

            output, _ = msg.payload
            outputs.append(output)

        output = gather(outputs, device=self.out_device)
        out_queue.get()

        return output

    def forward(self, input: TensorOrTensors) -> TensorOrTensors:
        in_queue, out_queue = self.spawn_workers()
        num_inputs = self.push_input(input, in_queue)
        return self.pull_output(num_inputs, in_queue, out_queue)
