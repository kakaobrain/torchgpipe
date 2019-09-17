"""Multithreading in pipeline parallelism."""
from contextlib import contextmanager
from queue import Queue
import sys
from threading import Thread
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Generator, List, Optional, Tuple, Type, Union, cast

import torch

from torchgpipe.microbatch import Batch
from torchgpipe.stream import AbstractStream, use_device, use_stream

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
                 stream: AbstractStream,
                 *,
                 compute: Callable[[], Batch],
                 finalize: Optional[Callable[[Batch], None]],
                 ) -> None:
        self.device = device
        self.stream = stream
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
