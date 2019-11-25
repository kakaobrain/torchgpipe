from weakref import WeakSet

import pytest
import torch
from torch import nn

from torchgpipe import GPipe
from torchgpipe.skip import pop, skippable, stash
from torchgpipe.skip.portal import Portal
from torchgpipe.skip.tracker import current_skip_tracker as _current_skip_tracker


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


def test_when_portal_tensor_is_deleted(monkeypatch):
    # Collect existing portals.
    portals = []

    def init(self, tensor, tensor_life, init=Portal.__init__):
        init(self, tensor, tensor_life)
        portals.append(self)
    monkeypatch.setattr(Portal, '__init__', init)

    # Store the total number of portals that hold a tensor at each PortalBlue.
    blue_timeline = []

    def blue(self, blue=Portal.blue):
        phony = blue(self)
        blue_timeline.append(len([p for p in portals if p.tensor is not None]))
        return phony
    monkeypatch.setattr(Portal, 'blue', blue)

    # Store the total number of portals that hold a tensor at each PortalOrange.
    orange_timeline = []

    def orange(self, phony, orange=Portal.orange):
        tensor = orange(self, phony)
        orange_timeline.append(len([p for p in portals if p.tensor is not None]))
        return tensor
    monkeypatch.setattr(Portal, 'orange', orange)

    model = nn.Sequential(Stash(), Pop())
    model = GPipe(model, balance=[1, 1], devices=['cpu', 'cpu'], chunks=3)
    input = torch.rand(3, requires_grad=True, device=model.devices[0])
    model(input).norm().backward()

    # The timelines are deterministic because this test uses CPUs only.
    assert blue_timeline == [
        1,  # PortalBlue[0]
        2,  # PortalBlue[1]
        3,  # PortalBlue[2] without checkpointing
        0,  # Recomputed PortalBlue[1]
        0,  # Recomputed PortalBlue[0]
    ]
    assert orange_timeline == [
        2,  # PortalOrange[0]
        3,  # PortalOrange[1]
        2,  # PortalOrange[2] without checkpointing
        1,  # Recomputed PortalOrange[1]
        0,  # Recomputed PortalOrange[0]
    ]
