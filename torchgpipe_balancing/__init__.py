"""A helper to roughly balance a sequential module.

Usage::

    import torch
    from torchgpipe import GPipe
    from torchgpipe_balancing import balance_by_time

    sample = torch.rand(128, 3, 224, 224)
    balance = balance_by_time(model, sample, partitions=4)

    gpipe = GPipe(model, balance, chunks=8)

"""
import time
from typing import List, Optional, Union

import torch
from torch import Tensor
import torch.nn as nn

from torchgpipe_balancing import utils

__all__ = ['balance_by_time', 'balance_by_size']


Device = Union[torch.device, int, str]


def balance_by_time(module: nn.Sequential,
                    sample: Tensor,
                    *,
                    partitions: int = 1,
                    device: Optional[Device] = None,
                    timeout: float = 1.0,
                    ) -> List[int]:
    """Balances the given seqeuntial module by elapsed time per layer.

    Args:
        module (nn.Sequential):
            sequential module to be partitioned
        sample (Tensor):
            example input

    Keyword Args:
        partitions (int):
            intended number of partitions (default: 1)
        device (torch.device):
            CUDA device where the module is profiled (default: any related CUDA
            device or ``torch.device('cuda')``)
        timeout (float):
            profiling iterates again if the timeout (as second) is not exceeded
            (default: 1 second)

    Returns:
        A list of number of layers in each partition. Use it for the
        ``balance`` parameter of :class:`torchgpipe.GPipe`.

    """
    sample, device = utils.concentrate_on_device(module, sample, device)

    times: List[List[float]] = [[] for _ in module]

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

                times[i].append(tock - tick)

    return utils.balance_cost(map(sum, times), partitions)


def balance_by_size(module: nn.Sequential,
                    sample: Tensor,
                    *,
                    partitions: int = 1,
                    device: Optional[Device] = None,
                    ) -> List[int]:
    """Balances the given seqeuntial module by memory usage per layer.

    Note:
        This function relies on :func:`torch.cuda.reset_max_memory_allocated`
        which is introduced at PyTorch 1.1. Therefore, it doesn't support
        neither CPU tensors nor PyTorch 1.0.x.

    Args:
        module (nn.Sequential):
            sequential module to be partitioned
        sample (Tensor):
            example input

    Keyword Args:
        partitions (int):
            intended number of partitions (default: 1)
        device (torch.device):
            CUDA device where the module is profiled (default: any related CUDA
            device or ``torch.device('cuda')``)

    Returns:
        A list of number of layers in each partition. Use it for the
        ``balance`` parameter of :class:`torchgpipe.GPipe`.

    """
    if not hasattr(torch.cuda, 'reset_max_memory_allocated'):
        raise NotImplementedError('balance_by_size requires PyTorch>=1.1')

    sample, device = utils.concentrate_on_device(module, sample, device)

    if device.type != 'cuda':
        raise ValueError('balance_by_size supports only CUDA device')

    sizes: List[int] = []

    x = sample
    with utils.training_sandbox(module):
        for i, layer in enumerate(module):
            torch.cuda.reset_max_memory_allocated(device)
            x = layer(x)
            sizes.append(torch.cuda.max_memory_allocated(device))

    return utils.balance_cost(sizes, partitions)
