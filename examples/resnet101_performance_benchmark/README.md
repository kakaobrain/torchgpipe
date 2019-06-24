# ResNet-101 Performance Benchmark

This example reproduces performance benchmark on ResNet-101, as stated by
reported in Figure 3(b) of the GPipe paper.

The benchmark cares of only training performance rather than the model's
accuracy. The batch size is adjusted to achieve higher throughput without any
large batch training tricks. This example also doesn't feed actual dataset like
ImageNet or CIFAR-100. Instead, a fake dataset with 50k 3×224×224 tensors is
used to eliminate data loading overhead.

Every experiment setting is optimized for Tesla P40 GPUs.

## Result

Experiment | Throughput          | Speed up
---------- | ------------------: | -------:
naive-1    | 100.506 samples/sec |   1.000x
pipeline-1 |  73.925 samples/sec |   0.736x
pipeline-2 | 135.691 samples/sec |   1.350x
pipeline-4 | 230.216 samples/sec |   2.291x
pipeline-8 | 312.945 samples/sec |   3.114x

## Optimized Environment

- Python 3.6.7
- PyTorch 1.1.0
- CUDA 9.0.176
- 8 Tesla P40 GPUs
- 8+ Intel E5-2650 v4 CPUs

## To Reproduce

First, resolve the dependencies. We highly recommend to use a separate virtual
environment only for this benchmark:

```sh
$ pip install -r requirements.txt
```

Then, run each benchmark:

```sh
$ python main.py naive-1
$ python main.py pipeline-1
$ python main.py pipeline-2
$ python main.py pipeline-4
$ python main.py pipeline-8
```
