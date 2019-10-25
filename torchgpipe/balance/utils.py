"""Internal utilities."""
from contextlib import contextmanager
from typing import Generator, List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from torchgpipe.balance import blockpartition

__all__: List[str] = []


Device = Union[torch.device, int, str]


def concentrate_on_device(module: nn.Sequential,
                          sample: Tensor,
                          device: Optional[Device],
                          ) -> Tuple[Tensor, torch.device]:
    """Moves the module and sample to the same CUDA device."""
    # The argument device is set.
    if device is not None:
        device = torch.device(device)
        module.to(device)
        return sample.to(device), torch.device(device)

    # The sample is on a CUDA device.
    if sample.is_cuda:
        module.to(sample.device)
        return sample, sample.device

    try:
        first_param = next(module.parameters())
    except StopIteration:
        pass
    else:
        # The first parameter is on a CUDA device.
        if first_param.is_cuda:
            device = first_param.device
            module.to(device)
            return sample.to(device), device

    # Move to the CUDA device without index by default.
    default_cuda = torch.device('cuda')
    module.to(default_cuda)
    return sample.to(default_cuda), default_cuda


@contextmanager
def training_sandbox(module: nn.Sequential) -> Generator[None, None, None]:
    """A context manager for training in sandbox mode."""
    training = module.training
    module.train()
    state_dict = {k: v.clone() for k, v in module.state_dict().items()}

    yield

    module.load_state_dict(state_dict)
    module.train(training)


def synchronize_device(device: torch.device) -> None:
    if device.type == 'cpu':
        return
    torch.cuda.synchronize(device)


def balance_cost(cost: List[int], partitions: int) -> List[int]:
    partitioned = blockpartition.solve(cost, partitions)
    return [len(p) for p in partitioned]
