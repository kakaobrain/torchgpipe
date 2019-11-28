"""A ResNet bottleneck implementation but using :class:`nn.Sequential`."""
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Tuple, Union

from torch import Tensor, nn

from torchgpipe.skip import Namespace, pop, skippable, stash

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


@skippable(stash=['identity'])
class Identity(nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:  # type: ignore
        yield stash('identity', tensor)
        return tensor


@skippable(pop=['identity'])
class Residual(nn.Module):
    """A residual block for ResNet."""

    def __init__(self, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.downsample = downsample

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        identity = yield pop('identity')
        if self.downsample is not None:
            identity = self.downsample(identity)
        return input + identity


def bottleneck(inplanes: int,
               planes: int,
               stride: int = 1,
               downsample: Optional[nn.Module] = None,
               inplace: bool = False,
               ) -> nn.Sequential:
    """Creates a bottleneck block in ResNet as a :class:`nn.Sequential`."""

    layers: NamedModules = OrderedDict()

    ns = Namespace()
    layers['identity'] = Identity().isolate(ns)  # type: ignore

    layers['conv1'] = conv1x1(inplanes, planes)
    layers['bn1'] = nn.BatchNorm2d(planes)
    layers['relu1'] = nn.ReLU(inplace=inplace)

    layers['conv2'] = conv3x3(planes, planes, stride)
    layers['bn2'] = nn.BatchNorm2d(planes)
    layers['relu2'] = nn.ReLU(inplace=inplace)

    layers['conv3'] = conv1x1(planes, planes * 4)
    layers['bn3'] = nn.BatchNorm2d(planes * 4)
    layers['residual'] = Residual(downsample).isolate(ns)  # type: ignore
    layers['relu3'] = nn.ReLU(inplace=inplace)

    return nn.Sequential(layers)
