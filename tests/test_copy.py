import pytest
import torch

from torchgpipe.copy import Copy, Wait
from torchgpipe.stream import (CPUStream, current_stream, get_device, is_cuda, new_stream,
                               use_stream)

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')


def _test_copy_wait(prev_stream, next_stream):
    device = get_device(prev_stream)

    with use_stream(prev_stream):
        if is_cuda(prev_stream):
            torch.cuda._sleep(100000000)
        x = torch.ones(100, device=device, requires_grad=True)

    y, = Copy.apply(prev_stream, next_stream, x)
    y = Wait.apply(prev_stream, next_stream, x)

    with use_stream(next_stream):
        assert torch.allclose(y.sum(), torch.tensor(100.0, device=device))
        y.norm().backward()
    with use_stream(prev_stream):
        assert torch.allclose(x.grad.sum(), torch.tensor(10.0, device=device))


def test_copy_wait_cpu_cpu():
    prev_stream = CPUStream
    next_stream = CPUStream
    _test_copy_wait(prev_stream, next_stream)


@skip_if_no_cuda
def test_copy_wait_cpu_cuda():
    prev_stream = CPUStream
    next_stream = current_stream(torch.device('cuda'))
    _test_copy_wait(prev_stream, next_stream)


@skip_if_no_cuda
def test_copy_wait_cuda_cpu():
    prev_stream = current_stream(torch.device('cuda'))
    next_stream = CPUStream
    _test_copy_wait(prev_stream, next_stream)


@skip_if_no_cuda
def test_copy_wait_cuda_cuda():
    prev_stream = current_stream(torch.device('cuda'))
    next_stream = new_stream(torch.device('cuda'))
    _test_copy_wait(prev_stream, next_stream)
