# ResNet-101 Speed Benchmark

This example reproduces speed benchmark on ResNet-101, as stated by reported in
Figure 3(b) of the GPipe paper.

The benchmark cares of only training speed rather than the model's accuracy.
The batch size is adjusted to achieve higher throughput without any large batch
training tricks. This example also doesn't feed actual dataset like ImageNet or
CIFAR-100. Instead, fake 3×224×224 tensors over 10 labels are used to eliminate
data loading overhead.

Every experiment setting is optimized for Tesla P40 GPUs.

## Result

Experiment | Throughput          | Speed up
---------- | ------------------: | -------:
naive-1    |  92.539 samples/sec |   1.000x
pipeline-1 |  69.960 samples/sec |   0.756x
pipeline-2 | 137.788 samples/sec |   1.489x
pipeline-4 | 243.322 samples/sec |   2.629x
pipeline-8 | 404.084 samples/sec |   4.367x

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

Then, run each benchmark:

```sh
$ python main.py naive-1
$ python main.py pipeline-1
$ python main.py pipeline-2
$ python main.py pipeline-4
$ python main.py pipeline-8
```
