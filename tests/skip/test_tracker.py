from queue import Queue
import threading

import pytest
import torch
from torch import nn

from torchgpipe.microbatch import Batch
from torchgpipe.skip import pop, skippable, stash
from torchgpipe.skip.tracker import SkipTracker, current_skip_tracker


def test_default_skip_tracker():
    q = Queue()

    def f():
        q.put(current_skip_tracker())

    t = threading.Thread(target=f)
    t.start()
    t.join()

    skip_tracker = q.get()

    assert isinstance(skip_tracker, SkipTracker)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
def test_default_skip_tracker_by_data_parallel():
    @skippable(stash=['foo'])
    class Stash(nn.Module):
        def forward(self, input):
            yield stash('foo', input)
            return input * 2

    @skippable(pop=['foo'])
    class Pop(nn.Module):
        def forward(self, input):
            foo = yield pop('foo')
            return foo

    model = nn.Sequential(Stash(), Pop())
    model = nn.DataParallel(model, device_ids=[0, 0], output_device=0)

    input = torch.rand(10, device=0)
    output = model(input)

    assert torch.allclose(output, input)


def test_reuse_portal():
    skip_tracker = SkipTracker()

    batch = Batch(torch.tensor([1.0]))
    a = torch.tensor([2.0])
    b = torch.tensor([2.0])

    skip_tracker._save_portal(batch, None, 'test', a)
    portal = skip_tracker.portals[(None, 'test')]

    assert portal.tensor is not None
    skip_tracker._save_portal(batch, None, 'test', b)
    assert portal.tensor is None
