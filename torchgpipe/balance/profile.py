"""Per-layer profilers."""
from itertools import chain
import time
from typing import List, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from torchgpipe.balance import utils
from torchgpipe.microbatch import Batch

__all__: List[str] = []


Device = Union[torch.device, int, str]

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]


def backward(layer: nn.Module, output: Batch) -> Batch:
    """Backpropagates a layer for profiling."""
    if any(p.grad is not None for p in layer.parameters()):
        raise ValueError('some parameter has gradient')

    tensors = tuple(y for y in output if y.requires_grad)
    if not tensors:
        return output

    torch.autograd.backward(tensors, tensors)

    # Free memory for gradients.
    for p in layer.parameters():
        p.grad = None

    # Detach from autograd graph.
    for y in output:
        requires_grad = y.requires_grad
        y.detach_().requires_grad_(requires_grad)

    return output


def profile_times(module: nn.Sequential,
                  sample: TensorOrTensors,
                  timeout: float,
                  ) -> List[int]:
    """Profiles elapsed times per layer."""
    time_bufs: List[List[float]] = [[] for _ in module]

    begun_at = time.time()
    while time.time() - begun_at < timeout:
        batch = Batch(sample)

        with utils.training_sandbox(module):
            for i, layer in enumerate(module):
                if batch[0].device.type == 'cuda':
                    torch.cuda.synchronize(batch[0].device)
                tick = time.time()

                batch = batch.call(layer)
                batch = backward(layer, batch)

                if batch[0].device.type == 'cuda':
                    torch.cuda.synchronize(batch[0].device)
                tock = time.time()

                time_bufs[i].append(tock - tick)

    us = 1_000_000
    return [sum(int(t*us) for t in buf) for buf in time_bufs]


def profile_sizes(module: nn.Sequential,
                  input: TensorOrTensors,
                  chunks: int,
                  param_scale: float,
                  ) -> List[int]:
    """Profiles CUDA memory usage per layer."""
    batch = Batch(input)
    sizes: List[int] = []

    tensors = chain(batch, module.parameters(), module.buffers())
    if any(x.device.type != 'cuda' for x in tensors):
        raise ValueError('size profiler supports only CUDA device')

    latent_scale = batch[0].size(0) / chunks
    for i, x in enumerate(batch):
        batch[i] = x[:1].clone()

    with utils.training_sandbox(module):
        for i, layer in enumerate(module):
            # Detect memory usage at both forward and backward, which means
            # that size of activations and activation gradients.
            device = batch[0].device
            torch.cuda.reset_max_memory_allocated(device)
            memory_before = torch.cuda.max_memory_allocated(device)

            batch = batch.call(layer)
            batch = backward(layer, batch)

            memory_after = torch.cuda.max_memory_allocated(device)
            latent_size = memory_after - memory_before

            # Analyze size of parameters.
            param_size = sum(p.storage().size() * p.storage().element_size()
                             for p in layer.parameters())

            # Combine size of parameters and activations with normalize scales.
            size = latent_size*latent_scale + param_size*param_scale
            sizes.append(int(size))

    return sizes
