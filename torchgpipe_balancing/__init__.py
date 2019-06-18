"""A helper to roughly balance a sequential module.

Usage::

    import torch
    from torchgpipe import GPipe
    from torchgpipe_balancing import balance_by_time

    sample = torch.rand(128, 3, 224, 224)
    balance = balance_by_time(model, sample, partitions=4)

    gpipe = GPipe(model, balance, chunks=8)

"""
from typing import Any

__all__ = ['balance_by_time', 'balance_by_size']


def balance_by_time(*args: Any, **kwargs: Any) -> Any:
    """Balances the given seqeuntial module by elapsed time per layer."""
    raise NotImplementedError


def balance_by_size(*args: Any, **kwargs: Any) -> Any:
    """Balances the given seqeuntial module by memory usage per layer."""
    raise NotImplementedError
