"""Per-layer profilers."""
import time
from typing import List, Optional, Union

import torch
from torch import Tensor
import torch.nn as nn

from torchgpipe.balancing import utils

__all__: List[str] = []


Device = Union[torch.device, int, str]


def profile_times(module: nn.Sequential,
                  sample: Tensor,
                  device: Optional[Device],
                  timeout: float,
                  ) -> List[int]:
    """Profiles elapsed times per layer."""
    sample, device = utils.concentrate_on_device(module, sample, device)

    time_bufs: List[List[float]] = [[] for _ in module]

    begun_at = time.time()
    while time.time() - begun_at < timeout:

        x = sample
        with utils.training_sandbox(module):
            for i, layer in enumerate(module):
                utils.synchronize_device(device)
                tick = time.time()

                x = layer(x)

                utils.synchronize_device(device)
                tock = time.time()

                time_bufs[i].append(tock - tick)

    us = 1_000_000
    return [sum(int(t*us) for t in buf) for buf in time_bufs]


def profile_sizes(module: nn.Sequential,
                  sample: Tensor,
                  device: Optional[Device],
                  ) -> List[int]:
    """Profiles CUDA memory usage per layer."""
    if not hasattr(torch.cuda, 'reset_max_memory_allocated'):
        raise NotImplementedError('balance_by_size requires PyTorch>=1.1')

    sample, device = utils.concentrate_on_device(module, sample, device)

    if device.type != 'cuda':
        raise ValueError('balance_by_size supports only CUDA device')

    sizes: List[int] = []

    x = sample
    with torch.cuda.device(device), utils.training_sandbox(module):
        for i, layer in enumerate(module):
            torch.cuda.reset_max_memory_allocated(device)

            size_before = torch.cuda.max_memory_allocated(device)
            x = layer(x)
            size_after = torch.cuda.max_memory_allocated(device)

            size = size_after - size_before
            sizes.append(size)

    return sizes
