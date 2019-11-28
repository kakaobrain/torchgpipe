from typing import Any, List

import torch
from torch import Tensor, nn

__all__: List[str] = []


class Operation(nn.Module):
    """Includes the operation name into the representation string for
    debugging.
    """

    def __init__(self, name: str, module: nn.Module):
        super().__init__()
        self.name = name
        self.module = module

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}[{self.name}]'

    def forward(self, *args: Any) -> Any:  # type: ignore
        return self.module(*args)


class FactorizedReduce(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=2, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        x = input
        x = self.relu(x)
        x = torch.cat([self.conv1(x), self.conv2(self.pad(x[:, :, 1:, 1:]))], dim=1)
        x = self.bn(x)
        return x


def none(channels: int, stride: int) -> Operation:
    module: nn.Module
    if stride == 1:
        module = nn.Identity()
    else:
        module = FactorizedReduce(channels, channels)
    return Operation('none', module)


def avg_pool_3x3(channels: int, stride: int) -> Operation:
    module = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    return Operation('avg_pool_3x3', module)


def max_pool_3x3(channels: int, stride: int) -> Operation:
    module = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    return Operation('max_pool_3x3', module)


def max_pool_2x2(channels: int, stride: int) -> Operation:
    module = nn.MaxPool2d(2, stride=stride, padding=0)
    return Operation('max_pool_2x2', module)


def conv_1x7_7x1(channels: int, stride: int) -> Operation:
    c = channels
    module = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(c, c//4, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(c//4),

        nn.ReLU(inplace=False),
        nn.Conv2d(c//4, c//4, kernel_size=(1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.BatchNorm2d(c//4),

        nn.ReLU(inplace=False),
        nn.Conv2d(c//4, c//4, kernel_size=(7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(c//4),

        nn.ReLU(inplace=False),
        nn.Conv2d(c//4, c, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(c),
    )
    return Operation('conv_1x7_7x1', module)


def conv_1x1(channels: int, stride: int) -> Operation:
    c = channels
    module = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(c, c, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(c),
    )
    return Operation('conv_1x1', module)


def conv_3x3(channels: int, stride: int) -> Operation:
    c = channels
    module = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(c, c//4, kernel_size=1, bias=False),
        nn.BatchNorm2d(c//4),

        nn.ReLU(inplace=False),
        nn.Conv2d(c//4, c//4, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c//4),

        nn.ReLU(inplace=False),
        nn.Conv2d(c//4, c, kernel_size=1, bias=False),
        nn.BatchNorm2d(c),
    )
    return Operation('conv_3x3', module)
