"""The pipeline parallelism of GPipe."""
from queue import Queue
from types import TracebackType
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Type, Union, cast

import torch
from torch import nn

from torchgpipe.checkpoint import Checkpointing
from torchgpipe.copy import Copy, Wait
from torchgpipe.dependency import Fork, Join
from torchgpipe.microbatch import Batch
from torchgpipe.stream import AbstractStream, CPUStream, current_stream
from torchgpipe.worker import Task, spawn_workers

__all__: List[str] = []


ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]

# Queue is generic only in stubs.
# https://mypy.readthedocs.io/en/latest/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
if TYPE_CHECKING:
    InQueue = Queue[Optional['Task']]
    OutQueue = Queue[Tuple[bool, Union[Tuple['Task', Batch], ExcInfo, None]]]
else:
    InQueue = Queue
    OutQueue = Queue


def depend(fork_from: Batch, join_to: Batch) -> None:
    fork_from[0], phony = Fork.apply(fork_from[0])
    join_to[0] = Join.apply(join_to[0], phony)


def copy(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Copy.apply(prev_stream, next_stream, *batch)


def wait(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[0] = Wait.apply(prev_stream, next_stream, batch[0])


def clock_cycles(n: int, m: int) -> Iterable[List[Tuple[int, int]]]:
    """Generates schedules for each clock cycle."""
    # i: index of partition
    # j: index of micro-batch
    #
    # k (i,j) (i,j) (i,j)
    # - ----- ----- -----
    # 0 (0,0)
    # 1 (0,1) (1,0)
    # 2 (0,2) (1,1) (2,0)
    # 3       (1,2) (2,1)
    # 4             (2,2)
    for k in range(n+m-1):
        yield [(i, k-i) for i in range(max(1+k-m, 0), min(1+k, n))]


class Pipeline:
    """The pipeline parallelism for GPipe."""

    def __init__(self,
                 batches: List[Batch],
                 partitions: List[nn.Sequential],
                 devices: Optional[List[torch.device]] = None,
                 copy_streams: Optional[List[List[AbstractStream]]] = None,
                 checkpoint_stop: int = 0,
                 ) -> None:
        self.batches = batches
        self.partitions = partitions

        if devices is None:
            devices = [torch.device('cpu') for _ in partitions]
        self.devices = devices

        if copy_streams is None:
            copy_streams = [[CPUStream] * len(batches) for _ in partitions]
        self.copy_streams = copy_streams

        self.checkpoint_stop = checkpoint_stop

    def run(self) -> None:
        """Runs pipeline parallelism.

        It modifies the given batches in place.

        """
        batches = self.batches
        partitions = self.partitions

        n = len(partitions)
        m = len(batches)

        with spawn_workers(n) as (in_queues, out_queues):
            for schedule in clock_cycles(n, m):
                self.fence(schedule)
                self.compute(schedule, in_queues, out_queues)

    def fence(self, schedule: List[Tuple[int, int]]) -> None:
        """Copies micro-batches after computation for the previous
        micro-batches.
        """
        batches = self.batches
        copy_streams = self.copy_streams

        for i, j in schedule:
            if j != 0:
                depend(batches[j-1], batches[j])

            if i != 0:
                prev_stream = copy_streams[i-1][j]
                next_stream = copy_streams[i][j]
                copy(batches[j], prev_stream, next_stream)

    def compute(self,
                schedule: List[Tuple[int, int]],
                in_queues: List[InQueue],
                out_queues: List[OutQueue],
                ) -> None:
        """Run tasks with synchronization to copy streams. A task consists of
        "compute" and "finalize". "compute" is executed on worker threads
        parallelly. "finalize" is executed when all worker threads complete to
        execute "compute".
        """
        batches = self.batches
        partitions = self.partitions
        devices = self.devices
        copy_streams = self.copy_streams
        checkpoint_stop = self.checkpoint_stop

        n = len(partitions)
        streams = [current_stream(d) for d in devices]
        exc_info: Optional[ExcInfo] = None

        for i, j in schedule:
            batch = batches[j]
            partition = partitions[i]
            device = devices[i]

            # 1. Synchronize the current stream with the copy stream.
            if i != 0:
                wait(batch, copy_streams[i][j], streams[i])

            # 2. Determine whether checkpointing or not.
            checkpoint = (j < checkpoint_stop)
            if checkpoint:
                chk = Checkpointing(partition, batch)
                task = Task(device, streams[i], compute=chk.checkpoint, finalize=chk.recompute)
                del chk

            else:
                def compute(batch: Batch = batch, partition: nn.Sequential = partition) -> Batch:
                    return batch.call(partition)
                task = Task(device, streams[i], compute=compute, finalize=None)
                del compute

            # 3. Compute tasks in parallel.
            in_queues[i].put(task)

        for i, j in schedule:
            ok, payload = out_queues[i].get()

            # Hold the first exception.
            if exc_info is not None:
                continue
            elif not ok:
                exc_info = cast(ExcInfo, payload)
                continue

            task, batch = cast(Tuple[Task, Batch], payload)

            # 4. Synchronize the copy stream with the current stream.
            if i != n-1:
                wait(batch, streams[i], copy_streams[i][j])

            # 5. Finalize tasks.
            #
            #    If checkpointing is enabled, here the recomputation is
            #    scheduled at backpropagation.
            #
            task.finalize(batch)

            batches[j] = batch

        # Fail at the first exception.
        if exc_info is not None:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
