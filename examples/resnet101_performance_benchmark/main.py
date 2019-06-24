"""ResNet-101 Performance Benchmark"""
import platform
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import click
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

from resnet import resnet101
from torchgpipe import GPipe

Stuffs = Tuple[nn.Module, int, List[torch.device]]  # (model, batch_size, devices)
Experiment = Callable[[nn.Module, List[int]], Stuffs]


class Experiments:

    @staticmethod
    def naive1(model: nn.Module, devices: List[int]) -> Stuffs:
        batch_size = 128
        device = devices[0]
        model.to(device)
        return model, batch_size, [torch.device(device)]

    @staticmethod
    def pipeline1(model: nn.Module, devices: List[int]) -> Stuffs:
        batch_size = 128
        chunks = 1
        balance = [304]

        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks, checkpoint='always')
        return model, batch_size, list(model.devices)

    @staticmethod
    def pipeline2(model: nn.Module, devices: List[int]) -> Stuffs:
        batch_size = 256
        chunks = 16
        balance = [115, 189]

        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks)
        return model, batch_size, list(model.devices)

    @staticmethod
    def pipeline4(model: nn.Module, devices: List[int]) -> Stuffs:
        batch_size = 1024
        chunks = 64
        balance = [39, 78, 97, 90]

        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks)
        return model, batch_size, list(model.devices)

    @staticmethod
    def pipeline8(model: nn.Module, devices: List[int]) -> Stuffs:
        batch_size = 1792
        chunks = 64
        balance = [22, 18, 27, 36, 36, 54, 54, 57]

        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks)
        return model, batch_size, list(model.devices)


EXPERIMENTS: Dict[str, Experiment] = {
    'naive-1': Experiments.naive1,
    'pipeline-1': Experiments.pipeline1,
    'pipeline-2': Experiments.pipeline2,
    'pipeline-4': Experiments.pipeline4,
    'pipeline-8': Experiments.pipeline8,
}


class RandomDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 50000

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        return torch.rand(3, 224, 224), random.randrange(10)


BASE_TIME: float = 0


def hr() -> None:
    """Prints a horizontal line."""
    width, _ = click.get_terminal_size()
    click.echo('-' * width)


def log(msg: str, clear: bool = False, nl: bool = True) -> None:
    """Prints a message with elapsed time."""
    if clear:
        # Clear the output line to overwrite.
        width, _ = click.get_terminal_size()
        click.echo('\b\r', nl=False)
        click.echo(' ' * width, nl=False)
        click.echo('\b\r', nl=False)

    t = time.time() - BASE_TIME
    h = t // 3600
    t %= 3600
    m = t // 60
    t %= 60
    s = t

    click.echo('%02d:%02d:%02d | ' % (h, m, s), nl=False)
    click.echo(msg, nl=nl)


def parse_devices(ctx: Any, param: Any, value: Optional[str]) -> List[int]:
    if value is None:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in value.split(',')]


@click.command()
@click.pass_context
@click.argument(
    'experiment',
    type=click.Choice(sorted(EXPERIMENTS.keys())),
)
@click.option(
    '--epochs', '-e',
    type=int,
    default=10,
    help='Number of epochs (default: 10)',
)
@click.option(
    '--skip-epochs', '-k',
    type=int,
    default=1,
    help='Number of epochs to skip in result (default: 1)',
)
@click.option(
    '--devices', '-d',
    metavar='0,1,2,3',
    callback=parse_devices,
    help='Device IDs to use (default: all CUDA devices)',
)
def cli(ctx: click.Context,
        experiment: str,
        epochs: int,
        skip_epochs: int,
        devices: List[int],
        ) -> None:
    """ResNet-101 Performance Benchmark"""
    if skip_epochs >= epochs:
        ctx.fail('--skip-epochs=%d must be less than --epochs=%d' % (skip_epochs, epochs))

    model: nn.Module = resnet101(num_classes=10)

    f = EXPERIMENTS[experiment]
    try:
        model, batch_size, _devices = f(model, devices)
    except ValueError as exc:
        # Examples:
        #   ValueError: too few devices to hold given partitions (devices: 1, paritions: 2)
        ctx.fail(str(exc))

    optimizer = SGD(model.parameters(), lr=0.1)

    in_device = _devices[0]
    out_device = _devices[-1]

    # This experiment cares about only training performance, rather than
    # accuracy. To eliminate any overhead due to data loading, we use a fake
    # dataset with random 224x224 images over 10 labels.
    dataset = RandomDataset()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )

    # HEADER ======================================================================================

    title = '%s, %d-%d epochs' % (experiment, skip_epochs+1, epochs)
    click.echo(title)
    click.echo('python: %s, torch: %s, cudnn: %s, cuda: %s, gpu: %s' % (
        platform.python_version(),
        torch.__version__,
        torch.backends.cudnn.version(),
        torch.version.cuda,
        torch.cuda.get_device_name(in_device)))

    # TRAIN =======================================================================================

    global BASE_TIME
    BASE_TIME = time.time()

    def run_epoch(epoch: int) -> Tuple[float, float]:
        torch.cuda.synchronize(in_device)
        tick = time.time()

        data_trained = 0
        for i, (input, target) in enumerate(loader):
            data_trained += len(input)

            input = input.to(in_device, non_blocking=True)
            target = target.to(out_device, non_blocking=True)

            output = model(input)
            loss = F.cross_entropy(output, target)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # 00:01:02 | 1/20 epoch (42%) | 200.000 samples/sec (estimated)
            percent = i / len(loader) * 100
            throughput = data_trained / (time.time()-tick)
            log('%d/%d epoch (%d%%) | %.3f samples/sec (estimated)'
                '' % (epoch+1, epochs, percent, throughput), clear=True, nl=False)

        torch.cuda.synchronize(in_device)
        tock = time.time()

        # 00:02:03 | 1/20 epoch | 200.000 samples/sec, 123.456 sec/epoch
        elapsed_time = tock - tick
        throughput = len(dataset) / elapsed_time
        log('%d/%d epoch | %.3f samples/sec, %.3f sec/epoch'
            '' % (epoch+1, epochs, throughput, elapsed_time), clear=True)

        return throughput, elapsed_time

    throughputs = []
    elapsed_times = []

    hr()
    for epoch in range(epochs):
        throughput, elapsed_time = run_epoch(epoch)

        if epoch < skip_epochs:
            continue

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)
    hr()

    # RESULT ======================================================================================

    # pipeline-4, 2-10 epochs | 200.000 samples/sec, 123.456 sec/epoch (average)
    n = len(throughputs)
    throughput = sum(throughputs) / n
    elapsed_time = sum(elapsed_times) / n
    click.echo('%s | %.3f samples/sec, %.3f sec/epoch (average)'
               '' % (title, throughput, elapsed_time))


if __name__ == '__main__':
    cli()
