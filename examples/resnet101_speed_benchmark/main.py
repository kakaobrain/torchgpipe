"""ResNet-101 Speed Benchmark"""
import platform
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import click
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD

from resnet import resnet101
from torchgpipe import GPipe

Stuffs = Tuple[nn.Module, int, List[torch.device]]  # (model, batch_size, devices)
Experiment = Callable[[nn.Module, List[int]], Stuffs]


class Experiments:

    @staticmethod
    def naive1(model: nn.Module, devices: List[int]) -> Stuffs:
        batch_size = 96
        device = devices[0]
        model.to(device)
        return model, batch_size, [torch.device(device)]

    @staticmethod
    def pipeline1(model: nn.Module, devices: List[int]) -> Stuffs:
        batch_size = 96
        chunks = 1
        balance = [370]

        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks, checkpoint='always')
        return model, batch_size, list(model.devices)

    @staticmethod
    def pipeline2(model: nn.Module, devices: List[int]) -> Stuffs:
        batch_size = 4096
        chunks = 256
        balance = [140, 230]

        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks)
        return model, batch_size, list(model.devices)

    @staticmethod
    def pipeline4(model: nn.Module, devices: List[int]) -> Stuffs:
        batch_size = 4096
        chunks = 256
        balance = [47, 96, 118, 109]

        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks)
        return model, batch_size, list(model.devices)

    @staticmethod
    def pipeline8(model: nn.Module, devices: List[int]) -> Stuffs:
        batch_size = 4096
        chunks = 64
        balance = [26, 22, 33, 44, 44, 66, 66, 69]

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
    """ResNet-101 Speed Benchmark"""
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

    # This experiment cares about only training speed, rather than accuracy. To
    # eliminate any overhead due to data loading, we use fake random 224x224
    # images over 10 labels.
    input = torch.rand(batch_size, 3, 224, 224, device=in_device)
    target = torch.randint(10, (batch_size,), device=out_device)

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
        steps = 50000 // batch_size
        for i in range(steps):
            data_trained += batch_size

            output = model(input)
            loss = F.cross_entropy(output, target)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # 00:01:02 | 1/20 epoch (42%) | 200.000 samples/sec (estimated)
            percent = i / steps * 100
            throughput = data_trained / (time.time()-tick)
            log('%d/%d epoch (%d%%) | %.3f samples/sec (estimated)'
                '' % (epoch+1, epochs, percent, throughput), clear=True, nl=False)

        torch.cuda.synchronize(in_device)
        tock = time.time()

        # 00:02:03 | 1/20 epoch | 200.000 samples/sec, 123.456 sec/epoch
        elapsed_time = tock - tick
        throughput = batch_size * steps / elapsed_time
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
