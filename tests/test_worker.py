import threading
import time

import pytest
import torch

from torchgpipe.microbatch import Batch
from torchgpipe.stream import CPUStream
from torchgpipe.worker import Task, spawn_workers


def test_join_running_workers():
    count = 0

    def counter():
        nonlocal count
        time.sleep(0.1)
        count += 1
        return Batch(())

    with spawn_workers(10) as (in_queues, out_queues):
        def call_in_worker(i, f):
            task = Task(torch.device('cpu'), CPUStream, compute=f, finalize=None)
            in_queues[i].put(task)

        for i in range(10):
            call_in_worker(i, counter)

    # There's no indeterminism because 'spawn_workers' joins all running
    # workers.
    assert count == 10


def test_join_running_workers_with_exception():
    class ExpectedException(Exception):
        pass

    count = 0

    def counter():
        nonlocal count
        time.sleep(0.1)
        count += 1
        return Batch(())

    with pytest.raises(ExpectedException):
        with spawn_workers(10) as (in_queues, out_queues):
            def call_in_worker(i, f):
                task = Task(torch.device('cpu'), CPUStream, compute=f, finalize=None)
                in_queues[i].put(task)

            for i in range(10):
                call_in_worker(i, counter)

            raise ExpectedException

    # There's no indeterminism because only 1 task can be placed in input
    # queues.
    assert count == 10


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


@pytest.mark.parametrize('grad_mode', [True, False])
def test_grad_mode(grad_mode):
    def detect_grad_enabled():
        x = torch.rand(1, requires_grad=torch.is_grad_enabled())
        return Batch(x)

    with torch.set_grad_enabled(grad_mode):
        with spawn_workers(1) as (in_queues, out_queues):
            task = Task(torch.device('cpu'), CPUStream, compute=detect_grad_enabled, finalize=None)
            in_queues[0].put(task)

            ok, (_, batch) = out_queues[0].get()

            assert ok
            assert batch[0].requires_grad == grad_mode
