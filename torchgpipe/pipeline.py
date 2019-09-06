"""The pipeline parallelism of GPipe."""
from contextlib import contextmanager
from functools import partial
from queue import Queue
import sys
from threading import Thread
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Generator, List, Optional, Tuple, Type, Union, cast

import torch
from torch import nn

from torchgpipe.checkpoint import Checkpointing
from torchgpipe.copy import Copy, Wait
from torchgpipe.dependency import Fork, Join
from torchgpipe.microbatch import Batch
from torchgpipe.stream import AbstractStream, current_stream, use_device, use_stream

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


def clock_cycles(n: int, m: int) -> Generator[List[Tuple[int, int]], None, None]:
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


def pipeline(batches: List[Batch],
             partitions: List[nn.Sequential],
             devices: List[torch.device],
             copy_streams: List[List[AbstractStream]],
             checkpoint_stop: int,
             ) -> None:
    """Runs pipeline parallelism.

    It modifies the given batches in place.

    """
    n = len(partitions)
    m = len(batches)

    exc_info: Optional[ExcInfo] = None
    streams = [current_stream(d) for d in devices]

    with spawn_workers(n) as (in_queues, out_queues):
        for schedule in clock_cycles(n, m):
            # Fence:
            #
            #   Copy micro-batches after computation for the previous
            #   micro-batches.
            #
            for i, j in schedule:
                if j != 0:
                    depend(batches[j-1], batches[j])

                # TODO(sublee): Replace "i-1" with the source partition index.
                if i != 0:
                    copy(batches[j], copy_streams[i-1][j], copy_streams[i][j])

            # Computation:
            #
            #   Run tasks with synchronization to copy streams. A task consists of
            #   "compute" and "finalize". "compute" is executed on worker threads
            #   parallelly. "finalize" is executed when all worker threads complete
            #   to execute "compute".
            #
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
                    task = Task(device, compute=chk.checkpoint, finalize=chk.recompute)
                    del chk
                else:
                    task = Task(device, compute=partial(batch.call, partition), finalize=None)

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


class Task:
    """A task represents how to compute a micro-batch on a partition.

    It consists of two parts: :meth:`compute` and :meth:`finalize`.
    :meth:`compute` should be executed in worker threads concurrently.
    :meth:`finalize` should be executed after when worker threads complete to
    execute :meth:`compute`.

    :meth:`compute` might be boosted by worker threads. Because it produces
    several CUDA API calls by user code. In PyTorch, parallel CUDA API calls
    are not serialized through GIL. So more than one CUDA API call can be
    produced at the same time.

    """

    def __init__(self,
                 device: torch.device,
                 compute: Callable[[], Batch],
                 finalize: Optional[Callable[[Batch], None]],
                 ) -> None:
        self.device = device
        self.stream = current_stream(device)
        self._compute = compute
        self._finalize = finalize

    def compute(self) -> Batch:
        with use_device(self.device), use_stream(self.stream):
            return self._compute()

    def finalize(self, batch: Batch) -> None:
        if self._finalize is None:
            return
        with use_device(self.device), use_stream(self.stream):
            self._finalize(batch)


def worker(in_queue: InQueue,
           out_queue: OutQueue,
           grad_enabled: bool,
           ) -> None:
    """The main loop of a worker thread."""
    torch.set_grad_enabled(grad_enabled)

    while True:
        task = in_queue.get()

        if task is None:
            out_queue.put((False, None))
            break

        try:
            batch = task.compute()
        except Exception:
            exc_info = cast(ExcInfo, sys.exc_info())
            out_queue.put((False, exc_info))
            continue

        out_queue.put((True, (task, batch)))


@contextmanager
def spawn_workers(count: int) -> Generator[Tuple[List[InQueue], List[OutQueue]], None, None]:
    """Spawns worker threads."""
    in_queues: List[InQueue] = []
    out_queues: List[OutQueue] = []

    grad_enabled = torch.is_grad_enabled()

    # Spwan workers.
    for _ in range(count):
        in_queue: InQueue = Queue(1)
        out_queue: OutQueue = Queue(1)

        t = Thread(target=worker, args=(in_queue, out_queue, grad_enabled))
        t.daemon = True
        t.start()

        in_queues.append(in_queue)
        out_queues.append(out_queue)

    try:
        yield (in_queues, out_queues)
    finally:
        # Close workers.
        for in_queue in in_queues:
            in_queue.put(None)
        for out_queue in out_queues:
            ok, payload = out_queue.get()
            assert not ok
            assert payload is None
