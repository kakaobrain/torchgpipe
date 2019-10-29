"""A helper to roughly balance a sequential module.

Usage::

    import torch
    from torchgpipe import GPipe
    from torchgpipe.balance import balance_by_time

    sample = torch.rand(128, 3, 224, 224)
    balance = balance_by_time(4, model, sample)

    gpipe = GPipe(model, balance, chunks=8)

"""
from typing import List, Union

import torch
from torch import Tensor
import torch.nn as nn

from torchgpipe.balance import utils
from torchgpipe.balance.profile import profile_sizes, profile_times

__all__ = ['balance_by_time', 'balance_by_size']


Device = Union[torch.device, int, str]


def balance_by_time(partitions: int,
                    module: nn.Sequential,
                    sample: Tensor,
                    *,
                    timeout: float = 1.0,
                    ) -> List[int]:
    """Naive automatic balancing by elapsed time per layer.

    Args:
        partitions (int):
            intended number of partitions
        module (nn.Sequential):
            sequential module to be partitioned
        sample (Tensor):
            example input with arbitrary batch size

    Keyword Args:
        timeout (float):
            profiling iterates again if the timeout (in second) is not exceeded
            (default: ``1.0``)

    Returns:
        A list of number of layers in each partition. Use it for the
        ``balance`` parameter of :class:`~torchgpipe.GPipe`.

    .. note::
        `module` and `sample` must be placed on the same device.

    """
    times = profile_times(module, sample, timeout)
    return utils.balance_cost(times, partitions)


def balance_by_size(partitions: int,
                    module: nn.Sequential,
                    sample: Tensor,
                    ) -> List[int]:
    """Naive automatic balancing by CUDA memory usage per layer.

    Args:
        partitions (int):
            intended number of partitions
        module (nn.Sequential):
            sequential module to be partitioned
        sample (Tensor):
            example input with arbitrary batch size

    Returns:
        A list of number of layers in each partition. Use it for the
        ``balance`` parameter of :class:`~torchgpipe.GPipe`.

    .. note::
        `module` and `sample` must be placed on the same CUDA device.

    """
    sizes = profile_sizes(module, sample)
    return utils.balance_cost(sizes, partitions)
