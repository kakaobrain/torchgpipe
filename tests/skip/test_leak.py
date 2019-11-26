from weakref import WeakSet

import pytest
import torch
from torch import nn

from torchgpipe import GPipe, is_checkpointing, is_recomputing
from torchgpipe.skip import pop, skippable, stash
from torchgpipe.skip.tracker import current_skip_tracker


@skippable(stash=['skip'])
class Stash(nn.Module):
    def forward(self, input):
        yield stash('skip', input)
        return input


@skippable(pop=['skip'])
class Pop(nn.Module):
    def forward(self, input):
        skip = yield pop('skip')
        return input + skip


@pytest.fixture
def count_leaked_portals(monkeypatch):
    """This fixture is a function to count leaked
    :class:`torchgpipe.skip.portal.Portal` objects.
    """
    skip_trackers = WeakSet()

    def count():
        counter = 0
        for skip_tracker in skip_trackers:
            counter += len([p for p in skip_tracker.portals if not (p.tensor is p.grad is None)])
            counter += len([x for x in skip_tracker.tensors if x is not None])
        return counter

    def current_skip_tracker():
        skip_tracker = _current_skip_tracker()
        skip_trackers.add(skip_tracker)
        return skip_tracker
    monkeypatch.setattr('torchgpipe.skip.tracker.current_skip_tracker', current_skip_tracker)

    return count


@pytest.mark.parametrize('train', [True, False], ids=['train', 'eval'])
@pytest.mark.parametrize('gpipe, checkpoint',
                         [(False, None), (True, 'never'), (True, 'always')],
                         ids=['bare', 'gpipe[never]', 'gpipe[always]'])
def test_skip_leak(count_leaked_portals, train, gpipe, checkpoint, monkeypatch):
    model = nn.Sequential(Stash(), Pop())

    if gpipe:
        model = GPipe(model,
                      balance=[1, 1],
                      devices=['cpu', 'cpu'],
                      chunks=2,
                      checkpoint=checkpoint)

    input = torch.rand(10, requires_grad=True)

    if train:
        model.train()
        model(input).norm().backward()
    else:
        model.eval()
        with torch.no_grad():
            model(input)

    assert count_leaked_portals() == 0


@pytest.mark.parametrize('train', [True, False], ids=['train', 'eval'])
@pytest.mark.parametrize('checkpoint', ['always', 'except_last', 'never'])
def test_delete_portal_tensor(train, checkpoint):
    # Without checkpointing:
    # +- Stash --+  +--- Pop ----+ - - - layers
    # | 2,blue,1 |--| 1,orange,0 | - - - tensor_life and portal function
    # +----------+  +------------+
    #
    # With checkpointing:
    # +- Stash --+  +--- Pop ----+  +--- Pop'----+  +- Stash'--+
    # | 3,blue,2 |--| 2,orange,1 |--| 1,orange,0 |--| 1,blue,0 |
    # +----------+  +------------+  +------------+  +----------+

    def portal_tensor_life_is(tensor_life, skip_tracker=None):
        if skip_tracker is None:
            skip_tracker = current_skip_tracker()

        # Get the current portal.
        portal = list(skip_tracker.portals.values())[0]

        if tensor_life == 0:
            return portal.tensor_life == 0 and portal.tensor is None
        else:
            return portal.tensor_life == tensor_life and portal.tensor is not None

    # Check the portal tensor after 'Stash'.
    stash_ = Stash()

    @stash_.register_forward_hook
    def check_portal_tensor_after_stash(*_):
        if is_checkpointing():
            assert portal_tensor_life_is(2)
        elif is_recomputing():
            assert portal_tensor_life_is(0)
        else:
            assert portal_tensor_life_is(1)

    pop_ = Pop()

    @pop_.register_forward_hook
    def check_portal_tensor_after_pop(*_):
        if is_checkpointing():
            assert portal_tensor_life_is(1)
        elif is_recomputing():
            assert portal_tensor_life_is(0)
        else:
            assert portal_tensor_life_is(0)

    class NoPortalTensorAtBackward(nn.Module):
        class F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.skip_tracker = current_skip_tracker()
                return input.detach()

            @staticmethod
            def backward(ctx, grad):
                assert portal_tensor_life_is(0, skip_tracker=ctx.skip_tracker)
                return grad

        def forward(self, input):
            return self.F.apply(input)

    model = nn.Sequential(NoPortalTensorAtBackward(), stash_, pop_)
    model = GPipe(model,
                  balance=[2, 1],
                  devices=['cpu', 'cpu'],
                  chunks=2,
                  checkpoint=checkpoint)

    input = torch.rand(10, requires_grad=True)

    if train:
        model.train()
        output = model(input)
        output.norm().backward()
    else:
        model.eval()
        with torch.no_grad():
            model(input)
