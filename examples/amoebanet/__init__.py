"""An AmoebaNet-D implementation but using :class:`nn.Sequential`. :func:`amoebanetd`
returns a :class:`nn.Sequential`.
"""
from collections import OrderedDict
from typing import Tuple

import torch
from torch import nn

from amoebanet.flatten_sequential import flatten_sequential
from amoebanet.genotype import Genotype, amoebanetd_genotype
from amoebanet.surgery import Concat, FirstAnd, InputOne, MergeTwo, Shift, Twin, TwinLast

__all__ = ['amoebanetd']


class Identity(nn.Module):
    """Through the tensor::

     x ---> x

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return x


class ReLUConvBN(nn.Module):
    """Operation ReLU + Conv2d + BatchNorm2d"""

    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int,
                 padding: int, affine: bool = True):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class FactorizedReduce(nn.Module):
    """Operation Factorized reduce"""

    def __init__(self, in_planes: int, out_planes: int, affine: bool = True):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(in_planes, out_planes // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(in_planes, out_planes // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


def skip_connect(channel: int, stride: int, affine: bool) -> nn.Module:
    if stride == 1:
        return Identity()
    return FactorizedReduce(channel, channel, affine=affine)


def avg_pool_3x3(channel: int, stride: int, affine: bool) -> nn.Sequential:
    return nn.Sequential(
        nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
        nn.Conv2d(channel, channel, (1, 1), stride=(1, 1), padding=(0, 0), bias=False),
        nn.BatchNorm2d(channel, affine=affine))


def max_pool_3x3(channel: int, stride: int, affine: bool) -> nn.Sequential:
    return nn.Sequential(
        nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
        nn.Conv2d(channel, channel, (1, 1), stride=(1, 1), padding=(0, 0), bias=False),
        nn.BatchNorm2d(channel, affine=affine))


def max_pool_2x2(channel: int, stride: int, affine: bool) -> nn.Sequential:
    return nn.Sequential(
        nn.MaxPool2d(2, stride=stride, padding=0),
        nn.Conv2d(channel, channel, (1, 1), stride=(1, 1), padding=(0, 0), bias=False),
        nn.BatchNorm2d(channel, affine=affine))


def conv_7x1_1x7(channel: int, stride: int, affine: bool) -> nn.Sequential:
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(channel, channel // 4, (1, 1), stride=(1, 1), padding=(0, 0), bias=False),
        nn.BatchNorm2d(channel // 4, affine=affine),
        nn.ReLU(inplace=False),
        nn.Conv2d(channel // 4, channel // 4, (1, 7),
                  stride=(1, stride), padding=(0, 3), bias=False),
        nn.BatchNorm2d(channel // 4, affine=affine),
        nn.ReLU(inplace=False),
        nn.Conv2d(channel // 4, channel // 4, (7, 1),
                  stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(channel // 4, affine=affine),
        nn.ReLU(inplace=False),
        nn.Conv2d(channel // 4, channel, (1, 1), stride=(1, 1), padding=(0, 0), bias=False),
        nn.BatchNorm2d(channel, affine=affine))


def conv_1x1(channel: int, stride: int, affine: bool) -> nn.Sequential:
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(channel, channel, (1, 1), stride=(stride, stride), padding=(0, 0), bias=False),
        nn.BatchNorm2d(channel, affine=affine))


def conv_3x3(channel: int, stride: int, affine: bool) -> nn.Sequential:
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(channel, channel // 4, (1, 1), stride=(1, 1), padding=(0, 0), bias=False),
        nn.BatchNorm2d(channel // 4, affine=affine),
        nn.ReLU(inplace=False),
        nn.Conv2d(channel // 4, channel // 4, (3, 3),
                  stride=(stride, stride), padding=(1, 1), bias=False),
        nn.BatchNorm2d(channel // 4, affine=affine),
        nn.ReLU(inplace=False),
        nn.Conv2d(channel // 4, channel, (1, 1), stride=(1, 1), padding=(0, 0), bias=False),
        nn.BatchNorm2d(channel, affine=affine))


OPS = {
    'skip_connect': skip_connect,
    'avg_pool_3x3': avg_pool_3x3,
    'max_pool_3x3': max_pool_3x3,
    'max_pool_2x2': max_pool_2x2,
    'conv_7x1_1x7': conv_7x1_1x7,
    'conv_1x1____': conv_1x1,
    'conv_3x3____': conv_3x3,
}


class Classifier(nn.Module):

    def __init__(self, channel_prev: int, num_classes: int):
        super().__init__()
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(channel_prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        s1 = self.global_pooling(x[1])
        y = self.classifier(s1.view(s1.size(0), -1))
        return y


def make_stem(channel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(3, channel, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(channel),
        Twin(),
    )


def make_cell(genotype: Genotype,
              channel_prev_prev: int, channel_prev: int, channel: int,
              reduction: bool, reduction_prev: bool
              ) -> Tuple[nn.Sequential, int]:

    preprocess0: nn.Module = nn.Sequential()

    if reduction_prev:
        preprocess0 = FactorizedReduce(channel_prev_prev, channel)
    elif channel_prev_prev != channel:
        preprocess0 = ReLUConvBN(channel_prev_prev, channel, 1, 1, 0)
    preprocess1 = ReLUConvBN(channel_prev, channel, 1, 1, 0)

    if reduction:
        op_names, indices = zip(*genotype.reduce)
        concat = genotype.reduce_concat
    else:
        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat

    ops = []
    for name, index in zip(op_names, indices):
        if reduction and index < 2:
            stride = 2
        else:
            stride = 1

        op = OPS[name](channel, stride, True)
        ops.append((op, index))

    layers = [
        InputOne(preprocess0, i=0),
        TwinLast(),
        InputOne(preprocess1, i=1),
        # Output: (preprocess0(x[0]), preprocess1(x[1]), x[1])
        # The last tensor x[1] is passed until the cell output.
        # The comments below call x[1] "skip".
    ]

    for i in range(len(ops) // 2):
        op0, i0 = ops[i*2]
        op1, i1 = ops[i*2 + 1]

        layers.extend([
            InputOne(op0, i=i0, insert=2+i),
            # Output: x..., op0(x[i0]), skip]

            InputOne(op1, i=i1, insert=2 + i + 1),
            # Output: x..., op0(x[i0]), op1(x[i1]), skip

            MergeTwo(2 + i, 2 + i + 1),
            # Output: x..., op0(x[i0]) + op1(x[i1]), skip
        ])

    layers.extend([
        # Move skip to the head.
        Shift(),
        # Output: skip, x...

        FirstAnd(Concat(concat)),
        # Output: skip, concat(x...)
    ])

    multiplier = len(concat)

    return nn.Sequential(*layers), multiplier


def amoebanetd(num_classes: int = 10,
               num_layers: int = 4,
               num_filters: int = 512,
               ) -> nn.Sequential:
    """Builds an AmoebaNet-D model for ImageNet."""
    channel = num_filters // 4

    def make_layer(channel: int,
                   num_layers: int,
                   genotype: Genotype,
                   ) -> Tuple[nn.Sequential, int]:
        n = num_layers
        channel_prev_prev, channel_prev, channel_curr = channel, channel, channel
        cells = []

        reduction_prev = False
        reduction = True
        channel_curr *= 2
        cell, multiplier = make_cell(genotype, channel_prev_prev,
                                     channel_prev, channel_curr, reduction, reduction_prev)
        channel_prev_prev, channel_prev = channel_prev, multiplier * channel_curr
        cells.append(cell)

        reduction_prev = True
        reduction = True
        channel_curr *= 2
        cell, multiplier = make_cell(genotype, channel_prev_prev,
                                     channel_prev, channel_curr, reduction, reduction_prev)
        channel_prev_prev, channel_prev = channel_prev, multiplier * channel_curr
        cells.append(cell)

        reduction = False
        reduction_prev = True
        for _ in range(n):
            cell, multiplier = make_cell(genotype, channel_prev_prev,
                                         channel_prev, channel_curr, reduction, reduction_prev)
            channel_prev_prev, channel_prev = channel_prev, multiplier * channel_curr
            cells.append(cell)
            reduction_prev = False

        reduction_prev = False
        reduction = True
        channel_curr *= 2
        cell, multiplier = make_cell(genotype, channel_prev_prev,
                                     channel_prev, channel_curr, reduction, reduction_prev)
        channel_prev_prev, channel_prev = channel_prev, multiplier * channel_curr
        cells.append(cell)

        reduction = False
        reduction_prev = True
        for _ in range(n):
            cell, multiplier = make_cell(genotype, channel_prev_prev,
                                         channel_prev, channel_curr, reduction, reduction_prev)
            channel_prev_prev, channel_prev = channel_prev, multiplier * channel_curr
            cells.append(cell)
            reduction_prev = False

        reduction_prev = False
        reduction = True
        channel_curr *= 2
        cell, multiplier = make_cell(genotype, channel_prev_prev,
                                     channel_prev, channel_curr, reduction, reduction_prev)
        channel_prev_prev, channel_prev = channel_prev, multiplier * channel_curr
        cells.append(cell)

        reduction = False
        reduction_prev = True
        for _ in range(n):
            cell, multiplier = make_cell(genotype, channel_prev_prev,
                                         channel_prev, channel_curr, reduction, reduction_prev)
            channel_prev_prev, channel_prev = channel_prev, multiplier * channel_curr
            cells.append(cell)
            reduction_prev = False

        return nn.Sequential(*cells), channel_prev

    cells, channel_prev = make_layer(channel, num_layers, amoebanetd_genotype)

    model = nn.Sequential(OrderedDict([
        ('stem', make_stem(channel)),
        ('cells', cells),
        ('fin', Classifier(channel_prev, num_classes))
    ]))

    return flatten_sequential(model)
