import weakref

import pytest
import torch

from torchgpipe.dependency import Fork, Join


def test_phony():
    x = torch.zeros(0, requires_grad=True)

    y, phony = Fork.apply(x)
    z, phony2 = Fork.apply(y)

    # Fork doesn't modify the given tensor.
    assert y.data_ptr() == x.data_ptr()

    # Phony tensors have no space.
    assert phony.size() == (0,)

    # Phony storages should be cached.
    assert phony2.data_ptr() == phony.data_ptr()

    # Phony tensors should not be cached.
    assert phony2 is not phony


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
def test_fork_join():
    logs = []

    class Log(torch.autograd.Function):
        @staticmethod
        def forward(ctx, number, tensor):
            ctx.number = number
            return tensor.detach()

        @staticmethod
        def backward(ctx, grad):
            logs.append(ctx.number)
            return None, grad

    a = torch.rand(1, device='cpu', requires_grad=True)
    b = torch.rand(1, device='cuda', requires_grad=True)

    a = Log.apply(1, a)

    a, phony = Fork.apply(a)
    b = Join.apply(a, phony)

    b = Log.apply(2, b)
    b = b.to('cpu')

    (a+b).backward()

    assert logs == [2, 1]


def test_fork_leak():
    leak = None

    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input

        @staticmethod
        def backward(ctx, grad):
            nonlocal leak
            leak = weakref.ref(ctx)
            return grad

    x = torch.rand(1, requires_grad=True)
    x = F.apply(x)
    x, phony = Fork.apply(x)
    x = Join.apply(x, phony)

    x.backward()
    del x, phony

    assert leak() is None
