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
