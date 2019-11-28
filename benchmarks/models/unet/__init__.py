"""Simplified U-Net"""
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Generator, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torchgpipe.skip import Namespace, pop, skippable, stash
from unet.flatten_sequential import flatten_sequential

if TYPE_CHECKING:
    NamedModules = OrderedDict[str, nn.Module]
else:
    NamedModules = OrderedDict


@skippable(stash=['skip'], pop=[])
class Stash(nn.Module):
    def forward(self, input: Tensor) -> Generator[stash, None, Tensor]:  # type: ignore
        yield stash('skip', input)
        return input


@skippable(stash=[], pop=['skip'])
class PopCat(nn.Module):
    def forward(self, input: Tensor) -> Generator[pop, Tensor, Tensor]:  # type: ignore
        skipped_input = yield pop('skip')

        skip_shape = skipped_input.shape[2:]
        input_shape = input.shape[2:]

        if input_shape != skip_shape:
            pad = [d2 - d1 for d1, d2 in zip(input_shape, skip_shape)]
            pad = sum([[0, p] for p in pad[::-1]], [])
            input = F.pad(input, pad=pad)

        output = torch.cat((input, skipped_input), dim=1)
        return output


def conv_dropout_norm_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    layers: NamedModules = OrderedDict()
    layers['conv'] = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    layers['dropout'] = nn.Dropout2d(p=0.1)
    layers['norm'] = nn.InstanceNorm2d(out_channels)
    layers['relu'] = nn.LeakyReLU(negative_slope=1e-2)
    return nn.Sequential(layers)


def stacked_convs(in_channels: int,
                  hidden_channels: int,
                  out_channels: int,
                  num_convs: int,
                  ) -> nn.Sequential:
    layers: List[nn.Module] = []

    if num_convs == 1:
        layers.append(conv_dropout_norm_relu(in_channels, out_channels))

    elif num_convs > 1:
        layers.append(conv_dropout_norm_relu(in_channels, hidden_channels))

        for _ in range(num_convs - 2):
            layers.append(conv_dropout_norm_relu(hidden_channels, hidden_channels))

        layers.append(conv_dropout_norm_relu(hidden_channels, out_channels))

    return nn.Sequential(*layers)


def unet(depth: int = 5,
         num_convs: int = 5,
         base_channels: int = 64,
         input_channels: int = 3,
         output_channels: int = 1,
         ) -> nn.Sequential:
    """Builds a simplified U-Net model."""
    # The U-Net structure
    encoder_channels = [{
        'in': input_channels if i == 0 else base_channels * (2 ** (i - 1)),
        'mid': base_channels * (2 ** i),
        'out': base_channels * (2 ** i),
    } for i in range(depth)]

    bottleneck_channels = [{
        'in': base_channels * (2 ** (depth - 1)),
        'mid': base_channels * (2 ** depth),
        'out': base_channels * (2 ** (depth - 1)),
    }]

    inverted_decoder_channels = [{
        'in': base_channels * (2 ** (i + 1)),
        'mid': int(base_channels * (2 ** (i - 1))),
        'out': int(base_channels * (2 ** (i - 1))),
    } for i in range(depth)]

    # Build cells.
    def cell(ch: Dict[str, int]) -> nn.Sequential:
        return stacked_convs(ch['in'], ch['mid'], ch['out'], num_convs)

    encoder_cells = [cell(c) for c in encoder_channels]
    bottleneck_cells = [cell(c) for c in bottleneck_channels]
    decoder_cells = [cell(c) for c in inverted_decoder_channels]

    # Link long skip connections.
    #
    # [ encoder ]--------------[ decoder ]--[ segment ]
    #    [ encoder ]--------[ decoder ]
    #       [ encoder ]--[ decoder ]
    #            [ bottleneck ]
    #
    namespaces = [Namespace() for _ in range(depth)]

    encoder_layers: List[nn.Module] = []
    for i in range(depth):
        ns = namespaces[i]
        encoder_layers.append(nn.Sequential(OrderedDict([
            ('encode', encoder_cells[i]),
            ('skip', Stash().isolate(ns)),  # type: ignore
            ('down', nn.MaxPool2d(2, stride=2))
        ])))
    encoder = nn.Sequential(*encoder_layers)

    bottleneck = nn.Sequential(*bottleneck_cells)

    decoder_layers: List[nn.Module] = []
    for i in reversed(range(depth)):
        ns = namespaces[i]
        decoder_layers.append(nn.Sequential(OrderedDict([
            ('up', nn.Upsample(scale_factor=2)),
            ('skip', PopCat().isolate(ns)),  # type: ignore
            ('decode', decoder_cells[i])
        ])))
    decoder = nn.Sequential(*decoder_layers)

    final_channels = inverted_decoder_channels[0]['out']
    segment = nn.Conv2d(final_channels, output_channels, kernel_size=1, bias=False)

    # Construct a U-Net model as nn.Sequential.
    model = nn.Sequential(OrderedDict([
        ('encoder', encoder),
        ('bottleneck', bottleneck),
        ('decoder', decoder),
        ('segment', segment)
    ]))
    model = flatten_sequential(model)
    return model
