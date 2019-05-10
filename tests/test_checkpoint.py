import pytest
import torch
import torch.cuda

from torchgpipe.checkpoint import checkpoint
from torchgpipe.gpipe import first

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')


@pytest.mark.parametrize('device', devices)
def test_linear_checkpoints(device):
    # Copied from https://github.com/pytorch/pytorch/pull/18568.
    timeline = []

    class Log(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, name):
            ctx.name = name
            timeline.append('%s:forward' % name)
            return x.detach()

        @staticmethod
        def backward(ctx, grad_output):
            name = ctx.name
            timeline.append('%s:backward' % name)
            return grad_output, None

    a = torch.rand(1, requires_grad=True, device=device)
    b = torch.rand(1, requires_grad=True, device=device)

    # Increase the next function sequence number.
    _ = a + 1 + 2 + 3 + 4 + 5

    a, recompute_a = checkpoint(lambda a: Log.apply(a, 'a'), a)
    b = first(b, recompute_a)  # This line fixes this case.
    b, _ = checkpoint(lambda b: Log.apply(b, 'b'), b)
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
