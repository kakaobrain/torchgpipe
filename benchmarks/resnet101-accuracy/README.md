# ResNet-101 Accuracy Benchmark

GPipe should be transparent not to introduce additional hyperparameter tuning.
To verify the transparency, we reproduced top-1 error rate of ResNet-101 on
ImageNet, as reported in Table 2(c) of [Accurate, Large Minibatch
SGD](https://arxiv.org/abs/1706.02677) by Goyal et al.

Every experiment setting is optimized for Tesla P40 GPUs.

## Result

Experiment                | GPUs | Batch size | Learning rate | Top-1 error (%) | Throughput | Speed up
------------------------- | ---: | ---------: | ------------: | --------------: |----------: | -------:
reference-256 ([paper][]) |    8 |        256 |           0.1 |      22.08±0.06 |        N/A |      N/A
reference-8k ([paper][])  |  256 |         8K |           3.2 |      22.36±0.09 |        N/A |      N/A
dataparallel-256          |    2 |        256 |           0.1 |      22.02±0.11 |  180.344/s |       1×
dataparallel-1k           |    8 |         1K |           0.4 |      22.04±0.24 |  606.916/s |   3.365×
dataparallel-4k           |    8 |         4K |           1.6 |   Out of memory |        N/A |      N/A
pipeline-256              |    2 |        256 |           0.1 |      21.99±0.13 |  117.432/s |   0.651×
pipeline-1k               |    8 |         1K |           0.4 |      22.24±0.19 |  294.739/s |   1.634×
pipeline-4k               |    8 |         4K |           1.6 |      22.13±0.09 |  378.746/s |   2.100×

## Optimized Environment

- torchgpipe 0.0.3
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
$ python main.py dataparallel-1k   # 8 GPUs required
$ python main.py gpipe-256         # 2 GPUs required
$ python main.py gpipe-1k          # 8 GPUs required
$ python main.py gpipe-4k          # 8 GPUs required
```
