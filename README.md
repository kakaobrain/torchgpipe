# torchgpipe

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
- PyTorch 1.0+

To use torchgpipe, install it via PyPI:

```sh
$ pip install torchgpipe
```

To train a module with GPipe, simply wrap it with `torchgpipe.GPipe`. Your
module must be `nn.Sequential` as GPipe will automatically break up the module
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

### ResNet-101 Performance Benchmark

Experiment | torchgpipe | GPipe (original)
---------- | ----: | ----:
naive-1    | 1     | 1
pipeline-1 | 0.74  | 0.8
pipeline-2 | 1.352 | 1.418
pipeline-4 | 2.181 | 2.182
pipeline-8 | 2.808 | 2.891

The table shows the reproduced performance benchmark on ResNet-101, as stated
by reported in Figure 3(b) of the paper.

Naive-1 indicates the baseline setting that ResNet-101 on a single device is
trained without GPipe. The speeds under other settings are measured relative to
the speed of naive-1 (which is considered as the unit speed). Pipeline-k means
k partitions with GPipe using k devices. Pipeline-1 is slower than naive-1
since it does not benefit from pipeline parallelism but has checkpointing
overhead.

### AmoebaNet-D Memory Benchmark

<table>
  <thead>
    <tr>
      <th rowspan="2">Experiment</th>
      <th colspan="2">naive-1</th>
      <th colspan="2">pipeline-1</th>
      <th colspan="2">pipeline-2</th>
      <th colspan="2">pipeline-4</th>
      <th colspan="2">pipeline-8</th>
    </tr>
    <tr align="center">
      <td>torchgpipe</td>
      <td>GPipe<br>(original)</td>
      <td>torchgpipe</td>
      <td>GPipe<br>(original)</td>
      <td>torchgpipe</td>
      <td>GPipe<br>(original)</td>
      <td>torchgpipe</td>
      <td>GPipe<br>(original)</td>
      <td>torchgpipe</td>
      <td>GPipe<br>(original)</td>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td>AmoebaNet-D (L, F)</td>
      <td colspan="2">(6, 208)</td>
      <td colspan="2">(6, 416)</td>
      <td colspan="2">(6, 544)</td>
      <td colspan="2">(12, 544)</td>
      <td colspan="2">(24, 512)</td>
    </tr>
    <tr align="center">
      <td># of Model Parameters</td>
      <td>90M</td>
      <td>82M</td>
      <td>358M</td>
      <td>318M</td>
      <td>613M</td>
      <td>542M</td>
      <td>1.16B</td>
      <td>1.05B</td>
      <td>2.01B</td>
      <td>1.80B</td>
    </tr>
    <tr align="center">
      <td>Total Peak Model Parameter Memory</td>
      <td>1.00GB</td>
      <td>1.05GB</td>
      <td>4.01GB</td>
      <td>3.80GB</td>
      <td>6.45GB</td>
      <td>6.45GB</td>
      <td>13.00GB</td>
      <td>12.53GB</td>
      <td>22.42GB</td>
      <td>24.62GB</td>
    </tr>
    <tr align="center">
      <td>Total Peak Activation Memory</td>
      <td>-</td>
      <td>6.26GB</td>
      <td>6.64GB</td>
      <td>3.46GB</td>
      <td>11.31GB</td>
      <td>8.11GB</td>
      <td>18.72GB</td>
      <td>15.21GB</td>
      <td>35.78GB</td>
      <td>26.24GB</td>
    </tr>
  </tbody>
</table>

It shows the better memory utilization of AmoebaNet-D with GPipe,
as stated in Table 1 of the paper. The size of an AmoebaNet-D
model is determined by two hyperparameters L and F which are proportional
to the number of layers and filters, respectively.

The difference between naive-1 and pipeline-1 indicates GPipe's
capability to leverage training a larger model. With 8 GPUs,
GPipe is capable of training a model which is 22 times larger compared
to the naive-1 setting.

## Notes

This project is functional, but the interface is not confirmed yet. All public
APIs are subject to change without warning until v0.1.0.

## Authors and Licensing

torchgpipe project is developed by [Heungsub Lee][] and [Myungryong Jeong][] at
[Kakao Brain][], with [Sungbin Lim][], [Chiheon Kim][], [Ildoo Kim][], and
[Woonhyuk Baek][]'s help. It is distributed under [Apache License
2.0](LICENSE).

[Kakao Brain]: https://kakaobrain.com/
[Heungsub Lee]: https://subl.ee/
[Myungryong Jeong]: https://github.com/mrJeong
[Sungbin Lim]: https://github.com/sungbinlim
[Chiheon Kim]: https://github.com/chiheonk
[Ildoo Kim]: https://github.com/ildoonet
[Woonhyuk Baek]: https://github.com/wbaek

## Citation

If you apply this library to any project and research, please cite our code:

```
@misc{torchgpipe,
  author       = {Kakao Brain},
  title        = {torchgpipe, {A} {GPipe} implementation in {PyTorch}},
  howpublished = {\url{https://github.com/kakaobrain/torchgpipe}},
  year         = {2019}
}
```
