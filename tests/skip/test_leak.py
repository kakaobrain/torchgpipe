from weakref import WeakSet

import pytest
import torch
from torch import nn

from torchgpipe import GPipe
from torchgpipe.skip import pop, skippable, stash
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
