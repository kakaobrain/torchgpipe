"""A ResNet implementation but using :class:`nn.Sequential`. :func:`resnet101`
returns a :class:`nn.Sequential` instead of ``ResNet``.

This code is transformed :mod:`torchvision.models.resnet`.

"""
from collections import OrderedDict
from typing import Any, List

import torch.nn as nn

from resnet.bottleneck import bottleneck
from resnet.flatten_sequential import flatten_sequential

__all__ = ['resnet101']


def build_resnet(layers: List[int],
                 num_classes: int = 1000,
                 inplace: bool = False
                 ) -> nn.Sequential:
    """Builds a ResNet as a simple sequential model.

    Note:
        The implementation is copied from :mod:`torchvision.models.resnet`.

    """
    inplanes = 64

    def make_layer(planes: int,
                   blocks: int,
                   stride: int = 1,
                   inplace: bool = False,
                   ) -> nn.Sequential:
        nonlocal inplanes

        downsample = None
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layers = []
        layers.append(bottleneck(inplanes, planes, stride, downsample, inplace))
        inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(bottleneck(inplanes, planes, inplace=inplace))

        return nn.Sequential(*layers)

    # Build ResNet as a sequential model.
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
        ('bn1', nn.BatchNorm2d(64)),
        ('relu', nn.ReLU()),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ('layer1', make_layer(64, layers[0], inplace=inplace)),
        ('layer2', make_layer(128, layers[1], stride=2, inplace=inplace)),
        ('layer3', make_layer(256, layers[2], stride=2, inplace=inplace)),
        ('layer4', make_layer(512, layers[3], stride=2, inplace=inplace)),

        ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
        ('flat', nn.Flatten()),
        ('fc', nn.Linear(512 * 4, num_classes)),
    ]))

    # Flatten nested sequentials.
    model = flatten_sequential(model)

    # Initialize weights for Conv2d and BatchNorm2d layers.
    # Stolen from torchvision-0.4.0.
    def init_weight(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            return

        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            return

    model.apply(init_weight)

    return model


def resnet101(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([3, 4, 23, 3], **kwargs)
