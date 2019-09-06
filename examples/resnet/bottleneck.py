"""A ResNet bottleneck implementation but using :class:`nn.Sequential`."""
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn

__all__ = ['bottleneck']

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

if TYPE_CHECKING:
    NamedModules = OrderedDict[str, nn.Module]
else:
    NamedModules = OrderedDict


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Twin(nn.Module):
    #     ┌──────┐
    # a --│ Twin │--> a
    #     │   '--│--> a
    #     └──────┘
    def forward(self,  # type: ignore
                tensor: Tensor,
                ) -> Tuple[Tensor, Tensor]:
        return tensor, tensor


class Gutter(nn.Module):
    #     ┌───────────┐
    # a --│ Gutter[M] │--> M(a)
    # b --│-----------│--> b
    #     └───────────┘
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self,  # type: ignore
                input_and_skip: Tuple[Tensor, Tensor],
                ) -> Tuple[Tensor, Tensor]:
        input, skip = input_and_skip
        output = self.module(input)
        return output, skip


class Residual(nn.Module):
    """A residual block for ResNet."""

    def __init__(self, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.downsample = downsample

    def forward(self,  # type: ignore
                input_and_identity: Tuple[Tensor, Tensor],
                ) -> Tensor:
        input, identity = input_and_identity
        if self.downsample is not None:
            identity = self.downsample(identity)
        return input + identity


def bottleneck(inplanes: int,
               planes: int,
               stride: int = 1,
               downsample: Optional[nn.Module] = None,
               ) -> nn.Sequential:
    """Creates a bottlenect block in ResNet as a :class:`nn.Sequential`."""
    layers: NamedModules = OrderedDict()
    layers['twin'] = Twin()

    layers['conv1'] = Gutter(conv1x1(inplanes, planes))
    layers['bn1'] = Gutter(nn.BatchNorm2d(planes))
    layers['relu1'] = Gutter(nn.ReLU())

    layers['conv2'] = Gutter(conv3x3(planes, planes, stride))
    layers['bn2'] = Gutter(nn.BatchNorm2d(planes))
    layers['relu2'] = Gutter(nn.ReLU())

    layers['conv3'] = Gutter(conv1x1(planes, planes * 4))
    layers['bn3'] = Gutter(nn.BatchNorm2d(planes * 4))
    layers['residual'] = Residual(downsample)
    layers['relu3'] = nn.ReLU()

    return nn.Sequential(layers)
