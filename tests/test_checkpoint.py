from collections import deque
from functools import partial

import pytest
import torch
import torch.cuda

from torchgpipe.checkpoint import Checkpoint, Checkpointing, Recompute
from torchgpipe.dependency import Fork, Join
from torchgpipe.microbatch import Batch

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')


@pytest.mark.parametrize('device', devices)
def test_serial_checkpoints(device):
    # Copied from https://github.com/pytorch/pytorch/pull/18568.
    timeline = []

    class Log(torch.autograd.Function):
        @staticmethod
        def forward(ctx, name, x):
            ctx.name = name
            timeline.append('%s:forward' % name)
            return x.detach()

        @staticmethod
        def backward(ctx, grad_output):
            name = ctx.name
            timeline.append('%s:backward' % name)
            return None, grad_output

    a = torch.rand(1, device=device)
    b = torch.rand(1, device=device)

    # Increase the next function sequence number.
    _ = a + 1 + 2 + 3 + 4 + 5

    phony = torch.zeros(0, device=device, requires_grad=True)

    a_recomputed = deque(maxlen=1)
    a_function = partial(Log.apply, 'a')
    a = Checkpoint.apply(phony, a_recomputed, a_function, True, a)
    a = Recompute.apply(phony, a_recomputed, a_function, True, a)

    a, a_phony = Fork.apply(a)
    b = Join.apply(b, a_phony)

    b_recomputed = deque(maxlen=1)
    b_function = partial(Log.apply, 'b')
    b = Checkpoint.apply(phony, b_recomputed, b_function, True, b)
    b = Recompute.apply(phony, b_recomputed, b_function, True, b)

    c = torch.cat((a, b))

    out = c.sum()

    #                        +--> {a} --Checkpoint(Log)--> {a}
    # {out} --Sum--> {c} --Cat     ^-----------------------------+
    #                        +--> {b} --Checkpoint(Log)--> {b} --First--> {b}
    out.backward()

    assert timeline == \
        ['a:forward', 'b:forward', 'b:forward', 'b:backward', 'a:forward', 'a:backward']
    #    |----------------------|  |-----------------------|  |-----------------------|
    #          forward pass            Checkpoint(Log[b])         Checkpoint(Log[a])


def test_not_requires_grad():
    x = Batch(torch.rand(1, requires_grad=False))
    assert not x[0].requires_grad

    def f(x):
        return x * 2

    chk = Checkpointing(f, x)
    x = chk.checkpoint()
    assert x[0].requires_grad

    chk.recompute(x)
    assert x[0].requires_grad

    x.tensor.backward()


def test_not_requires_grad_with_parameter():
    x = Batch(torch.rand(1, requires_grad=False))
    assert not x[0].requires_grad

    a = torch.rand(1, requires_grad=True)
    assert a.requires_grad

    def f(x):
        return x * a

    chk = Checkpointing(f, x)
    x = chk.checkpoint()
    chk.recompute(x)

    x.tensor.backward()
    assert a.grad is not None
