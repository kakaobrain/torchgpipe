# ResNet-101 Accuracy Benchmark

This example reproduces accuracy benchmark on ResNet-101, as stated by
reported in Table 2(c) of the [Accurate, Large Minibatch SGD][] paper.

Every experiment setting is optimized for Tesla P40 GPUs.

## Result

Experiment              | num gpus | kn  | learning rate | top-1 error (%) | throughput (samples/sec) | speed up
----------------------- | --------:| ---:|-------------: | ---------------:|-------------------------:|---------:
reference-256 [paper][] |        8 | 256 |           0.1 |   22.08&pm;0.06 |                      N/A |      N/A
reference-8k [paper][]  |      256 |  8k |           3.2 |   22.36&pm;0.09 |                      N/A |      N/A
dataparallel-256        |        2 | 256 |           0.1 |   22.02&pm;0.11 |                  180.344 |   1.000x
dataparallel-1k         |        8 |  1k |           0.4 |   22.04&pm;0.24 |                  606.916 |   3.365x
dataparallel-4k         |        8 |  4k |           1.6 |             OOM |                      N/A |      N/A
pipeline-256            |        2 | 256 |           0.1 |   21.99&pm;0.13 |                  117.432 |   0.651x
pipeline-1k             |        8 |  1k |           0.4 |   22.24&pm;0.19 |                  294.739 |   1.634x
pipeline-4k             |        8 |  4k |           1.6 |   22.13&pm;0.09 |                  378.746 |   2.100x



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
$ python main.py --devices 0,1 dataparallel  # 256
$ python main.py dataparallel  # 1k
$ python main.py gpipe-2-256   # gpipie 256
$ python main.py gpipe-8       # gpipie 1k
$ python main.py gpipe-8-4k    # gpipie 4k
```


[Accurate, Large Minibatch SGD]: https://arxiv.org/abs/1706.02677
[paper]: https://arxiv.org/abs/1706.02677
