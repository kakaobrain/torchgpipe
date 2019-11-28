"""U-Net Timeline Benchmark"""
import platform
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import click
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import SGD

from gpu_utils import track_gpu_utils
import torchgpipe
from torchgpipe import GPipe
from tuplify_skips import tuplify_skips
from unet import unet

Experiment = Callable[[nn.Sequential, List[int], List[int], int], GPipe]


class Experiments:

    @staticmethod
    def baseline(model: nn.Sequential,
                 balance: List[int],
                 devices: List[int],
                 chunks: int,
                 ) -> GPipe:
        import torchgpipe.pipeline
        torchgpipe.pipeline.depend = lambda *args: None
        model = tuplify_skips(model)
        gpipe = GPipe(model, balance, devices=devices, chunks=chunks)
        gpipe._copy_streams = [[torch.cuda.default_stream(d)] * chunks for d in gpipe.devices]
        return gpipe

    @staticmethod
    def dep_x_x(model: nn.Sequential,
                balance: List[int],
                devices: List[int],
                chunks: int,
                ) -> GPipe:
        # Disable portals.
        model = tuplify_skips(model)
        gpipe = GPipe(model, balance, devices=devices, chunks=chunks)

        # Disable streams.
        gpipe._copy_streams = [[torch.cuda.default_stream(d)] * chunks for d in gpipe.devices]
        return gpipe

    @staticmethod
    def dep_str_x(model: nn.Sequential,
                  balance: List[int],
                  devices: List[int],
                  chunks: int,
                  ) -> GPipe:
        # Disable portals.
        model = tuplify_skips(model)
        return GPipe(model, balance, devices=devices, chunks=chunks)

    @staticmethod
    def dep_str_ptl(model: nn.Sequential,
                    balance: List[int],
                    devices: List[int],
                    chunks: int,
                    ) -> GPipe:
        return GPipe(model, balance, devices=devices, chunks=chunks)


EXPERIMENTS: Dict[str, Experiment] = {
    'baseline': Experiments.baseline,
    'dep-x-x': Experiments.dep_x_x,
    'dep-str-x': Experiments.dep_str_x,
    'dep-str-ptl': Experiments.dep_str_ptl,
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
    """U-Net Timeline Benchmark"""
    if skip_epochs >= epochs:
        ctx.fail('--skip-epochs=%d must be less than --epochs=%d' % (skip_epochs, epochs))

    model = unet(depth=5, num_convs=5, base_channels=64, input_channels=3, output_channels=1)
    balance = [34, 76, 70, 61]
    chunks = 8

    f = EXPERIMENTS[experiment]
    try:
        gpipe = f(model, balance, devices, chunks)
    except ValueError as exc:
        # Examples:
        #   ValueError: too few devices to hold given partitions (devices: 1, paritions: 2)
        ctx.fail(str(exc))

    optimizer = SGD(model.parameters(), lr=0.1)

    in_device = gpipe.devices[0]
    out_device = gpipe.devices[-1]
    torch.cuda.set_device(in_device)

    side = 192
    steps = 16
    batch_size = 128

    input = torch.rand(batch_size, 3, side, side, device=in_device)
    target = torch.ones(batch_size, 1, side, side, device=out_device)
    data = [(input, target)] * steps

    # HEADER ======================================================================================

    title = f'{experiment}, {skip_epochs+1}-{epochs} epochs'
    click.echo(title)
    click.echo('torchgpipe: %s, python: %s, torch: %s, cudnn: %s, cuda: %s, gpu: %s' % (
        torchgpipe.__version__,
        platform.python_version(),
        torch.__version__,
        torch.backends.cudnn.version(),
        torch.version.cuda,
        torch.cuda.get_device_name(in_device)))

    # TRAIN =======================================================================================

    global BASE_TIME
    BASE_TIME = time.time()

    def run_epoch(epoch: int) -> Tuple[float, float, List[float]]:
        with track_gpu_utils([d.index for d in gpipe.devices]) as gpu_utils:
            torch.cuda.synchronize(in_device)
            tick = time.time()

            data_trained = 0
            for i, (input, target) in enumerate(data):
                data_trained += input.size(0)

                output = gpipe(input)
                output = cast(Tensor, output)
                loss = F.binary_cross_entropy_with_logits(output, target)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                # 00:01:02 | 1/20 epoch (42%) | 200.000 samples/sec (estimated)
                percent = (i+1) / len(data) * 100
                throughput = data_trained / (time.time()-tick)
                log('%d/%d epoch (%d%%) | %.3f samples/sec (estimated)'
                    '' % (epoch+1, epochs, percent, throughput), clear=True, nl=False)

            torch.cuda.synchronize(in_device)
            tock = time.time()

        # 00:02:03 | 1/20 epoch | 200.000 samples/sec, 123.456 sec/epoch | 100.0%, 96%, 89%, 100%
        elapsed_time = tock - tick
        throughput = data_trained / elapsed_time
        _gpu_utils = ', '.join([f'{u:.0%}' for u in gpu_utils])
        _gpu_utils += f' (total {sum(gpu_utils)/len(gpu_utils):.0%})'

        log('%d/%d epoch | %.3f samples/sec, %.3f sec/epoch | %s'
            '' % (epoch+1, epochs, throughput, elapsed_time, _gpu_utils), clear=True)

        return throughput, elapsed_time, gpu_utils

    throughputs = []
    elapsed_times = []
    gpu_timeline = []

    hr()
    for epoch in range(epochs):
        throughput, elapsed_time, gpu_utils = run_epoch(epoch)

        if epoch < skip_epochs:
            continue

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)
        gpu_timeline.append(gpu_utils)
    hr()

    # RESULT ======================================================================================

    # optimal, 2-10 epochs
    click.echo(title)

    # speed  | 53.006 samples/sec, 38.637 sec/epoch (average)
    n = len(throughputs)
    throughput = sum(throughputs) / n
    elapsed_time = sum(elapsed_times) / n
    click.echo('speed  | ', nl=False)
    click.echo(f'{throughput:.3f} samples/sec, {elapsed_time:.3f} sec/epoch (average)')

    # gpu%   | 59%, 71%, 74%, 64% (total 67%)
    gpu_utils = [sum(t)/len(t) for t in zip(*gpu_timeline)]
    click.echo('gpu%   | ', nl=False)
    click.echo(', '.join([f'{u:.0%}' for u in gpu_utils]), nl=False)
    click.echo(f' (total {sum(gpu_utils)/len(gpu_utils):.0%})')

    # memory | 17301 MB, 13650 MB, 7262 MB, 18759 MB (total 56972 MB)
    memory_usages: List[int] = []
    for d in gpipe.devices:
        memory_usages.append(torch.cuda.max_memory_cached(d))
    click.echo('memory | ', nl=False)
    click.echo(', '.join(f'{m/1024/1024/1024:.1f} GiB' for m in memory_usages), nl=False)
    click.echo(f' (total {sum(memory_usages)/1024/1024/1024:.1f} GiB)')


if __name__ == '__main__':
    cli()
