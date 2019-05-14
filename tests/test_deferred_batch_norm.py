from itertools import chain

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from torchgpipe.batchnorm import patch_deferred_batch_norm


@pytest.fixture
def bn():
    torch.manual_seed(0)
    return nn.BatchNorm2d(3)


@pytest.fixture
def def_bn():
    torch.manual_seed(0)
    module = nn.BatchNorm2d(3)
    module.apply(patch_deferred_batch_norm)
    return module


def tilt_dist(input):
    # Tilt variance by channel.
    rgb = input.transpose(0, 1)
    rgb[0] *= 1
    rgb[1] *= 10
    rgb[2] *= 100

    # Tilt mean by single batch.
    for i, single in enumerate(input):
        single += 10**i

    return input


def chunked_forward(model, input):
    output_chunks = []

    for chunk in input.chunk(4):
        output_chunks.append(model(chunk))

    return torch.cat(output_chunks)


def test_running_stats(bn, def_bn):
    input = torch.rand(16, 3, 224, 224)
    input = tilt_dist(input)

    bn(input)
    y = chunked_forward(def_bn, input)
    y.sum().backward()  # flush buffer

    assert torch.allclose(bn.running_mean, def_bn.running_mean, atol=1e-4)
    assert torch.allclose(bn.running_var, def_bn.running_var, atol=1e-4)


def test_noop():
    bn = nn.BatchNorm2d(3, track_running_stats=False)
    bn.apply(patch_deferred_batch_norm)
    y = bn(torch.rand(16, 3, 224, 224))
    y.mean().backward()


def test_eval(bn, def_bn):
    input = torch.rand(16, 3, 224, 224)
    input = tilt_dist(input)

    bn(input)
    y = chunked_forward(def_bn, input)
    y.sum().backward()  # flush buffer

    bn.eval()
    def_bn.eval()

    assert torch.allclose(bn(input), def_bn(input), atol=1e-4)


def test_backward(def_bn):
    input = torch.rand(16, 3, 224, 224)
    input = tilt_dist(input)

    output = chunked_forward(def_bn, input)

    # Should not raise this error:
    #
    #   RuntimeError: one of the variables needed for gradient computation has
    #   been modified by an inplace operation
    #
    output.sum().backward()


def test_optimize(bn, def_bn):
    opt = optim.SGD(chain(bn.parameters(), def_bn.parameters()), lr=0.1)

    for i in range(5):
        input = torch.rand(16, 3, 224, 224)
        input = tilt_dist(input)

        # train
        y = bn(input)
        a = y.sum()
        a.backward()

        y = chunked_forward(def_bn, input)
        b = y.sum()
        b.backward()

        opt.step()

        # eval
        bn.eval()
        def_bn.eval()

        with torch.no_grad():
            assert torch.allclose(bn(input), def_bn(input), atol=1e-1 * (10**i))


def test_conv_bn():
    torch.manual_seed(0)
    bn = nn.Sequential(
        nn.Conv2d(3, 3, 1),
        nn.BatchNorm2d(3),
    )

    torch.manual_seed(0)
    def_bn = nn.Sequential(
        nn.Conv2d(3, 3, 1),
        nn.BatchNorm2d(3),
    )
    def_bn.apply(patch_deferred_batch_norm)

    opt = optim.SGD(chain(bn.parameters(), def_bn.parameters()), lr=0.1)

    input = torch.rand(16, 3, 224, 224)
    input = tilt_dist(input)

    # 1st step
    a = bn(input)
    b = chunked_forward(def_bn, input)

    # Outputs are different. (per-mini-batch vs. per-micro-batch)
    assert not torch.allclose(a, b)

    a.sum().backward()
    b.sum().backward()
    opt.step()
    opt.zero_grad()

    # Conv layers are also trained differently because of their different outputs.
    assert not torch.allclose(bn[0].weight, def_bn[0].weight)

    # But BNs track identical running stats.
    assert torch.allclose(bn[1].running_mean, def_bn[1].running_mean, atol=1e+8)
    assert torch.allclose(bn[1].running_var, def_bn[1].running_var, atol=1e+22)

    # 2nd step
    a = bn(input)
    b = chunked_forward(def_bn, input)
    a.sum().backward()
    b.sum().backward()

    # BNs can't track identical running stats due to the different conv layers.
    assert not torch.allclose(bn[1].running_mean, def_bn[1].running_mean, atol=1e+8)
    assert not torch.allclose(bn[1].running_var, def_bn[1].running_var, atol=1e+22)


def test_input_requiring_grad(def_bn):
    input = torch.rand(16, 3, 224, 224, requires_grad=True)
    input = tilt_dist(input)

    chunked_forward(def_bn, input)

    assert not def_bn.sum.requires_grad
    assert def_bn.sum.grad_fn is None
