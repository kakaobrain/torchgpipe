import pytest
import torch
from torch import nn

from torchgpipe import GPipe


def test_python_autograd_function():
    # A Python autograd function might fail with this error:
    #
    #   RuntimeError: Returning Variables sharing storage with other Variables
    #   that require grad is not supported in Python functions. Please submit a
    #   feature request if you hit this error.
    #
    # It doesn't look like an essential restriction. But it happens on the
    # current PyTorch version. To avoid it, we should detach the tensor before
    # returning by identity autograd functions, such as Wait, Fork, and Join.
    #
    class Identity(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input

        @staticmethod
        def backward(ctx, grad):
            return grad

    class M(nn.Module):
        def forward(self, input):
            return Identity.apply(input)

    model = nn.Sequential(M(), M())
    model = GPipe(model, [1, 1], devices=['cpu', 'cpu'], checkpoint='always')

    x = torch.rand(42)
    y = model(x)
    assert torch.allclose(x, y)


def test_exception_no_hang():
    # In v0.0.2, once a failed partition receives a normal message
    # (non-closing) for the next micro-batch, a hang occured. The reason was
    # that a failed partition didn't call in_queue.task_done() on a normal
    # message. So the former partition was blocked at out_queue.join() for the
    # next of next micro-batch.
    class ExpectedException(Exception):
        pass

    class Pass(nn.Module):
        def forward(self, x):
            return x

    class Raise(nn.Module):
        def forward(self, x):
            raise ExpectedException()

    model = nn.Sequential(Pass(), Pass(), Raise())
    model = GPipe(model, [1, 1, 1], devices=['cpu', 'cpu', 'cpu'], chunks=3)

    with pytest.raises(ExpectedException):
        model(torch.rand(3))
