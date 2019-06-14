import pytest
import torch
import torch.cuda

from torchgpipe.microbatch import check, gather, scatter


def test_check():
    check(torch.tensor(42))
    check((torch.tensor(4), torch.tensor(2)))

    with pytest.raises(TypeError):
        check(42)

    with pytest.raises(TypeError):
        check('str')

    with pytest.raises(TypeError):
        check((torch.tensor(4), 2))


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
def test_default_device_index():
    default_cuda = torch.device('cuda')
    assert default_cuda.index is None

    x = torch.rand(2, 1)
    a, b = scatter(x, chunks=2, device=default_cuda)
    y = gather([a, b], device=default_cuda)

    assert a.is_cuda
    assert b.is_cuda
    assert y.is_cuda
