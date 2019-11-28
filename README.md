# torchgpipe <img src="docs/_static/not-pipe.svg" height="20" />

[![PyPI](https://img.shields.io/pypi/v/torchgpipe.svg)](https://pypi.org/project/torchgpipe)
[![Build Status](https://travis-ci.org/kakaobrain/torchgpipe.svg?branch=master)](https://travis-ci.org/kakaobrain/torchgpipe)
[![Coverage Status](https://coveralls.io/repos/github/KakaoBrain/torchgpipe/badge.svg?branch=master)](https://coveralls.io/github/KakaoBrain/torchgpipe?branch=master)
[![Documentation Status](https://readthedocs.org/projects/torchgpipe/badge/?version=latest)](https://torchgpipe.readthedocs.io/en/latest/?badge=latest)
[![Korean README](https://img.shields.io/badge/readme-korean-blue.svg)](README.ko.md)

A [GPipe](https://arxiv.org/abs/1811.06965) implementation in PyTorch. It is
optimized for CUDA rather than TPU.

```python
from torchgpipe import GPipe
model = nn.Sequential(a, b, c, d)
model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)
output = model(input)
```

## What is GPipe?

GPipe is a scalable pipeline parallelism library published by Google Brain,
which allows efficient training of large, memory-consuming models. According to
the paper, GPipe can train a 25x larger model by using 8x devices (TPU), and
train a model 3.5x faster by using 4x devices.

[GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)

Google trained AmoebaNet-B with 557M parameters over GPipe. This model has
achieved 84.3% top-1 and 97.0% top-5 accuracy on ImageNet classification
benchmark (the state-of-the-art performance as of May 2019).

GPipe uses (a) pipeline parallelism and (b) automatic recomputation of the
forward propagation during the backpropagation, hence leverages training a
large model. We refer to (b) as [checkpointing][], following the well-known
terminology in PyTorch community.

[checkpointing]: https://pytorch.org/docs/stable/checkpoint.html

<dl>
<dt>Pipeline Parallelism</dt>
<dd>GPipe splits a model into multiple partitions and places each partition on
    a different device to occupy more memory capacity. And it splits a
    mini-batch into multiple micro-batches to make the partitions work as
    parallel as possible.</dd>

<dt>Checkpointing</dt>
<dd>Checkpointing is applied to each partition to minimize the overall memory
    consumption by a model. During forward propagation, only the tensors at the
    boundaries between partitions are remembered. All other intermediate
    tensors are volatilized, and recomputed during backpropagation when
    necessary.</dd>
</dl>

## Usage

Currently, torchgpipe requires the following environments:

- Python 3.6+
- PyTorch 1.1+

To use torchgpipe, install it via PyPI:

```sh
$ pip install torchgpipe
```

To train a module with GPipe, simply wrap it with `torchgpipe.GPipe`. Your
module must be `nn.Sequential` as GPipe will automatically split the module
into partitions with consecutive layers. `balance` argument determines the
number of layers in each partition. `chunks` argument specifies the number of
micro-batches. Input, output, and intermediate tensors must be `Tensor` or
`Tuple[Tensor, ...]`.

The below example code shows how to split a module with four layers into four
partitions each having a single layer. This code also splits a mini-batch into
8 micro-batches:

```python
from torchgpipe import GPipe

model = nn.Sequential(a, b, c, d)
model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)

for input in data_loader:
    output = model(input)
```

## Documentation

Visit [torchgpipe.readthedocs.io][rtd] for more information including the API
references.

[rtd]: https://torchgpipe.readthedocs.io/

## Benchmarking

The full details and more benchmarks are available in
[torchgpipe.readthedocs.io][rtd-benchmarks].

[rtd-benchmarks]: https://torchgpipe.readthedocs.io/en/stable/benchmarks.html

### ResNet-101 Accuracy Benchmark

Batch size | torchgpipe | nn.DataParallel | Goyal et al.
---------- | ---------: | --------------: | -----------:
256        | 21.99±0.13 |      22.02±0.11 |   22.08±0.06
1K         | 22.24±0.19 |      22.04±0.24 |          N/A
4K         | 22.13±0.09 |             N/A |          N/A

GPipe should be transparent not to introduce additional hyperparameter tuning.
To verify the transparency, we reproduced top-1 error rate of ResNet-101 on
ImageNet, as reported in Table 2(c) of [Accurate, Large Minibatch
SGD](https://arxiv.org/abs/1706.02677) by Goyal et al.

### U-Net (B, C) Memory Benchmark

Experiment | U-Net (B, C) | Parameters | Memory usage
---------- | ------------ | ---------: | -----------:
baseline   | (6, 72)      |     362.2M |     20.3 GiB
pipeline-1 | (11, 128)    |      2.21B |     20.5 GiB
pipeline-2 | (24, 128)    |      4.99B |     43.4 GiB
pipeline-4 | (24, 160)    |      7.80B |     79.1 GiB
pipeline-8 | (48, 160)    |     15.82B |    154.1 GiB

The table shows how GPipe facilitates scaling U-Net models. *baseline* denotes
the baseline without pipeline parallelism nor checkpointing, and *pipeline-1*,
*-2*, *-4*, *-8* denotes that the model is trained with GPipe with the
corresponding number of partitions.

Here we used a simplified U-Net architecture. The size of a model is determined
by hyperparameters B and C which are proportional to the number of layers and
filters, respectively.

### U-Net (5, 64) Speed Benchmark

Experiment | Throughput | Speed up
---------- | ---------: | -------:
baseline   |   28.500/s |       1×
pipeline-1 |   24.456/s |   0.858×
pipeline-2 |   35.502/s |   1.246×
pipeline-4 |   67.042/s |   2.352×
pipeline-8 |   88.497/s |   3.105×

To verify efficiency with skip connections, we measured the throughput of U-Net
with various number of devices. We chose to use U-Net since it has several long
skip connections.

### AmoebaNet-D (18, 256) Speed Benchmark

Experiment | Throughput | torchgpipe | Huang et al.
---------- | ---------: | ---------: | -----------:
n=2, m=1   |   26.733/s |         1× |           1×
n=2, m=4   |   41.133/s |     1.546× |        1.07×
n=2, m=32  |   47.386/s |     1.780× |        1.21×
n=4, m=1   |   26.827/s |     1.006× |        1.13×
n=4, m=4   |   44.543/s |     1.680× |        1.26×
n=4, m=32  |   72.412/s |     2.711× |        1.84×
n=8, m=1   |   24.918/s |     0.932× |        1.38×
n=8, m=4   |   70.065/s |     2.625× |        1.72×
n=8, m=32  |  132.413/s |     4.966× |        3.48×

(*n*: number of partitions, *m*: number of micro-batches)

The table shows the reproduced speed benchmark on AmoebaNet-D (18, 256), as
reported in Table 2 of [GPipe](https://arxiv.org/abs/1811.06965) by Huang et
al. Note that we replaced *K* in the paper with *n*.

## Notes

This project is functional, but the interface is not confirmed yet. All public
APIs are subject to change without warning until v0.1.0.

## Authors and Licensing

torchgpipe project is developed by [Heungsub Lee][], [Myungryong Jeong][], and
[Chiheon Kim][] at [Kakao Brain][], with [Sungbin Lim][], [Ildoo Kim][],
[Woonhyuk Baek][], and [Boogeon Yoon][]'s help. It is distributed under [Apache
License 2.0](LICENSE).

[Kakao Brain]: https://kakaobrain.com/
[Heungsub Lee]: https://subl.ee/
[Myungryong Jeong]: https://github.com/mrJeong
[Chiheon Kim]: https://github.com/chiheonk
[Sungbin Lim]: https://github.com/sungbinlim
[Ildoo Kim]: https://github.com/ildoonet
[Woonhyuk Baek]: https://github.com/wbaek
[Boogeon Yoon]: https://github.com/bgyoon

## Citation

If you apply this library to any project and research, please cite our code:

```
@misc{torchgpipe,
  author       = {Lee, Heungsub and Jeong, Myungryong and Kim, Chiheon and
                  Lim, Sungbin and Kim, Ildoo and Baek, Woonhyuk and Yoon, Boogeon},
  title        = {torchgpipe, {A} {GPipe} implementation in {PyTorch}},
  howpublished = {\url{https://github.com/kakaobrain/torchgpipe}},
  year         = {2019}
}
```
