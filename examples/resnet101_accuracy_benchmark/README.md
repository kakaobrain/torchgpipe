# ResNet-101 Accuracy Benchmark

This example reproduces accuracy benchmark on ResNet-101, as stated by
reported in Table 2(c) of the [Accurate, Large Minibatch SGD][paper] paper.

Every experiment setting is optimized for Tesla P40 GPUs.

## Result

Experiment                | GPUs | Batch size | Learning rate | Top-1 error (%) | Throughput          | Speedup
------------------------- | ---: | ---------: | ------------: | --------------: |-------------------: | ------:
reference-256 ([paper][]) |    8 |        256 |           0.1 |      22.08±0.06 |                 N/A |     N/A
reference-8k ([paper][])  |  256 |         8k |           3.2 |      22.36±0.09 |                 N/A |     N/A
dataparallel-256          |    2 |        256 |           0.1 |      22.02±0.11 | 180.344 samples/sec |  1.000x
dataparallel-1k           |    8 |         1k |           0.4 |      22.04±0.24 | 606.916 samples/sec |  3.365x
dataparallel-4k           |    8 |         4k |           1.6 |   Out of memory |                 N/A |     N/A
pipeline-256              |    2 |        256 |           0.1 |      21.99±0.13 | 117.432 samples/sec |  0.651x
pipeline-1k               |    8 |         1k |           0.4 |      22.24±0.19 | 294.739 samples/sec |  1.634x
pipeline-4k               |    8 |         4k |           1.6 |      22.13±0.09 | 378.746 samples/sec |  2.100x

## Optimized Environment

- Python 3.6.9
- PyTorch 1.2.0
- CUDA 10.0.130
- 8 Tesla P40 GPUs
- 8+ Intel E5-2650 v4 CPUs

## To Reproduce

First, resolve the dependencies. We highly recommend to use a separate virtual
environment only for this benchmark:

```sh
$ pip install -r requirements.txt
```

Prepare ImageNet dataset at `./imagenet`:

```sh
$ python -c "import torchvision; torchvision.datasets.ImageNet('./imagenet', split='train', download=True)"
$ python -c "import torchvision; torchvision.datasets.ImageNet('./imagenet', split='val', download=True)"
```

Then, run each benchmark:

```sh
$ python main.py naive-128
$ python main.py dataparallel-256  # 2 GPUs required
$ python main.py dataparallel-1k   # 4 GPUs required
$ python main.py gpipe-256         # 2 GPUs required
$ python main.py gpipe-1k          # 8 GPUs required
$ python main.py gpipe-4k          # 8 GPUs required
```

[paper]: https://arxiv.org/abs/1706.02677
