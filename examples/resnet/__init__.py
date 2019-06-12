"""A ResNet implementation but using :class:`nn.Sequential`. :func:`resnet101`
returns a :class:`nn.Sequential` instead of ``ResNet``.

This code is transformed :mod:`torchvision.models.resnet`.

"""
from collections import OrderedDict
from typing import Any, List

from torch import Tensor
import torch.nn as nn

from resnet.bottleneck import bottleneck
from resnet.flatten_sequential import flatten_sequential

__all__ = ['resnet101']


class Flat(nn.Module):
    """Flattens any input tensor into an 1-d tensor."""

    def forward(self, x: Tensor):  # type: ignore
        return x.view(x.size(0), -1)


def build_resnet(layers: List[int],
                 num_classes: int = 1000,
                 ) -> nn.Sequential:
    """Builds a ResNet as a simple sequential model.

    Note:
        The implementation is copied from :mod:`torchvision.models.resnet`.

    """
    inplanes = 64

    def make_layer(planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        nonlocal inplanes

        downsample = None
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layers = []
        layers.append(bottleneck(inplanes, planes, stride, downsample))
        inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(bottleneck(inplanes, planes))

        return nn.Sequential(*layers)

    # Build ResNet as a sequential model.
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
        ('bn1', nn.BatchNorm2d(64)),
        ('relu', nn.ReLU()),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ('layer1', make_layer(64, layers[0])),
        ('layer2', make_layer(128, layers[1], stride=2)),
        ('layer3', make_layer(256, layers[2], stride=2)),
        ('layer4', make_layer(512, layers[3], stride=2)),

        ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
        ('flat', Flat()),
        ('fc', nn.Linear(512 * 4, num_classes)),
    ]))

    # Flatten nested sequentials.
    model = flatten_sequential(model)

    # Initialize weights for Conv2d and BatchNorm2d layers.
    def init_weight(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            assert isinstance(m.kernel_size, tuple)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

            m.weight.requires_grad = False
            m.weight.normal_(0, 2. / n**0.5)
            m.weight.requires_grad = True

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.weight.fill_(1)
            m.weight.requires_grad = True

            m.bias.requires_grad = False
            m.bias.zero_()
            m.bias.requires_grad = True

    model.apply(init_weight)

    return model


def resnet101(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([3, 4, 23, 3], **kwargs)
