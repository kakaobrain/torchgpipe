from collections import OrderedDict
from typing import Iterator, Tuple

from torch import nn


def flatten_sequential(module: nn.Sequential) -> nn.Sequential:
    """flatten_sequentials a nested sequential module."""
    if not isinstance(module, nn.Sequential):
        raise TypeError('not sequential')

    return nn.Sequential(OrderedDict(_flatten_sequential(module)))


def _flatten_sequential(module: nn.Sequential) -> Iterator[Tuple[str, nn.Module]]:
    for name, child in module.named_children():
        # flatten_sequential child sequential layers only.
        if isinstance(child, nn.Sequential):
            for sub_name, sub_child in _flatten_sequential(child):
                yield ('%s_%s' % (name, sub_name), sub_child)
        else:
            yield (name, child)
