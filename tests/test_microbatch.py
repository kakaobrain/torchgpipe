import torch

from torchgpipe.microbatch import gather, scatter


def test_gather_tensors():
    a = torch.zeros(1, 1)
    b = torch.zeros(1, 1)

    ab = gather([a, b], device=torch.device('cpu'))

    assert ab.size() == (2, 1)


def test_gather_tuples():
    a = (torch.zeros(1, 1), torch.zeros(2, 2))
    b = (torch.zeros(1, 1), torch.zeros(2, 2))

    ab = gather([a, b], device=torch.device('cpu'))

    assert isinstance(ab, tuple)
    assert ab[0].size() == (2, 1)
    assert ab[1].size() == (4, 2)


def test_scatter_tensor():
    ab = torch.zeros(2, 1)

    a, b = scatter(ab, chunks=2, device=torch.device('cpu'))

    assert a.size() == (1, 1)
    assert b.size() == (1, 1)


def test_scatter_tuple():
    ab = (torch.zeros(2, 1), torch.zeros(4, 2))

    a, b = scatter(ab, chunks=2, device=torch.device('cpu'))

    assert isinstance(a, tuple)
    assert isinstance(b, tuple)
    assert a[0].size() == (1, 1)
    assert b[0].size() == (1, 1)
    assert a[1].size() == (2, 2)
    assert b[1].size() == (2, 2)
