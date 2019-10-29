"""Per-layer profilers."""
from itertools import chain
import time
from typing import List, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from torchgpipe.balance import utils
from torchgpipe.microbatch import Batch

__all__: List[str] = []


Device = Union[torch.device, int, str]

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]


def profile_times(module: nn.Sequential,
                  sample: TensorOrTensors,
                  timeout: float,
                  ) -> List[int]:
    """Profiles elapsed times per layer."""
    time_bufs: List[List[float]] = [[] for _ in module]

    begun_at = time.time()
    while time.time() - begun_at < timeout:
        batch = Batch(sample)

        with utils.training_sandbox(module):
            for i, layer in enumerate(module):
                if batch[0].device.type == 'cuda':
                    torch.cuda.synchronize(batch[0].device)
                tick = time.time()

                batch = batch.call(layer)

                if batch[0].device.type == 'cuda':
                    torch.cuda.synchronize(batch[0].device)
                tock = time.time()

                time_bufs[i].append(tock - tick)

    us = 1_000_000
    return [sum(int(t*us) for t in buf) for buf in time_bufs]


def profile_sizes(module: nn.Sequential,
                  sample: TensorOrTensors,
                  ) -> List[int]:
    """Profiles CUDA memory usage per layer."""
    batch = Batch(sample)
    sizes: List[int] = []

    tensors = chain(batch, module.parameters(), module.buffers())
    if any(x.device.type != 'cuda' for x in tensors):
        raise ValueError('size profiler supports only CUDA device')

    with utils.training_sandbox(module):
        for i, layer in enumerate(module):
            # Detect memory usage at both forward and backward, which means
            # that size of activations and activation gradients.
            device = batch[0].device
            torch.cuda.reset_max_memory_allocated(device)
            memory_before = torch.cuda.max_memory_allocated(device)

            batch = batch.call(layer)

            memory_after = torch.cuda.max_memory_allocated(device)
            size = memory_after - memory_before
            sizes.append(int(size))

    return sizes
