import torch
from torch import nn

from torchgpipe import GPipe


def test_simple_linears():
    def sum_grad(parameters):
        return sum([p.grad.sum() for p in parameters if p.grad is not None])

    def zero_grad(parameters):
        for p in parameters:
            p.grad = None

    inputs = torch.rand(8, 1)
    model = nn.Sequential(
        nn.Linear(1, 2),
        nn.Linear(2, 4),
        nn.Linear(4, 2),
        nn.Linear(2, 1),
    )

    # Without GPipe
    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()

    grad_without_gpipe = sum_grad(model.parameters())

    zero_grad(model.parameters())

    # With GPipe
    model = GPipe(model, [2, 2], devices=['cpu', 'cpu'], chunks=4)

    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()

    grad_with_gpipe = sum_grad(model.parameters())

    # Both grads should be identical.
    assert torch.allclose(grad_with_gpipe, grad_without_gpipe)
