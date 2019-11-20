import pytest
import torch

from torchgpipe.dependency import fork, join
from torchgpipe.skip.portal import Portal
from torchgpipe.stream import default_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
def test_copy_returns_on_next_device():
    portal = Portal(torch.rand(1))

    prev_stream = default_stream(torch.device('cpu'))
    next_stream = default_stream(torch.device('cuda'))

    phony = torch.zeros(0, requires_grad=True)
    assert phony.device.type == 'cpu'

    phony = portal.copy(prev_stream, next_stream, phony)
    assert phony.device.type == 'cuda'


def test_blue_orange():
    tensor1 = torch.rand(1, requires_grad=True)
    tensor2 = torch.rand(1, requires_grad=True)

    # Same with: output = tensor1*2 + tensor2
    #
    #                +----------------------+
    #                |                      |
    # tensor2 -- PortalBlue -+      +- PortalOrange -+
    #                        |      |                |
    # tensor1 ------------ Join -- Fork --- Mul --- Add -- output
    #
    main = tensor1
    portal = Portal(tensor2)
    phony = portal.blue()
    main = join(main, phony)
    main, phony = fork(main)
    sub = portal.orange(phony)
    output = main*2 + sub

    output.backward()

    assert torch.allclose(tensor1.grad, torch.tensor([2.]))
    assert torch.allclose(tensor2.grad, torch.tensor([1.]))


def test_blue_orange_not_requires_grad():
    tensor1 = torch.rand(1, requires_grad=True)
    tensor2 = torch.rand(1)

    # Same with: output = tensor1*2 + tensor2
    #
    #                +----------------------+
    #                |                      |
    # tensor2 -- PortalBlue -+      +- PortalOrange -+
    #                        |      |                |
    # tensor1 ------------ Join -- Fork --- Mul --- Add -- output
    #
    main = tensor1
    portal = Portal(tensor2)
    phony = portal.blue()
    main = join(main, phony)
    main, phony = fork(main)
    sub = portal.orange(phony)
    output = main*2 + sub

    output.backward()

    assert torch.allclose(tensor1.grad, torch.tensor([2.]))
    assert tensor2.grad is None
