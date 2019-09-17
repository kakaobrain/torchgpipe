import threading

import torch

from torchgpipe.microbatch import Batch
from torchgpipe.pipeline import Task, spawn_workers
from torchgpipe.stream import CPUStream


def test_compute_multithreading():
    """Task.compute should be executed on multiple threads."""
    thread_ids = set()

    def log_thread_id():
        thread_id = threading.current_thread().ident
        thread_ids.add(thread_id)
        return Batch(())

    with spawn_workers(2) as (in_queues, out_queues):
        t = Task(torch.device('cpu'), CPUStream, compute=log_thread_id, finalize=None)
        for i in range(2):
            in_queues[i].put(t)
        for i in range(2):
            out_queues[i].get()

    assert len(thread_ids) == 2


def test_compute_success():
    """Task.compute returns (True, (task, batch)) on success."""
    def _42():
        return Batch(torch.tensor(42))

    with spawn_workers(1) as (in_queues, out_queues):
        t = Task(torch.device('cpu'), CPUStream, compute=_42, finalize=None)
        in_queues[0].put(t)
        ok, (task, batch) = out_queues[0].get()

        assert ok
        assert task is t
        assert isinstance(batch, Batch)
        assert batch[0].item() == 42


def test_compute_exception():
    """Task.compute returns (False, exc_info) on failure."""
    def zero_div():
        0/0

    with spawn_workers(1) as (in_queues, out_queues):
        t = Task(torch.device('cpu'), CPUStream, compute=zero_div, finalize=None)
        in_queues[0].put(t)
        ok, exc_info = out_queues[0].get()

        assert not ok
        assert isinstance(exc_info, tuple)
        assert issubclass(exc_info[0], ZeroDivisionError)
