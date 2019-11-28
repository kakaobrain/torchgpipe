# ResNet-101 Speed Benchmark

This example reproduces speed benchmark on ResNet-101, as reported in Figure
3(b) of [the fourth version](https://arxiv.org/abs/1811.06965v4) of GPipe by
Huang et al.

The benchmark cares of only training speed rather than the model's accuracy.
The batch size is adjusted to achieve higher throughput without any large batch
training tricks. This example also doesn't feed the actual ImageNet dataset.
Instead, fake 3×224×224 tensors over 1000 labels are used to eliminate data
loading overhead.

Every experiment setting is optimized for Tesla P40 GPUs.

## Result

Experiment | Throughput          | torchgpipe | Huang et al.
---------- | ------------------: | ---------: | -----------:
baseline   |  95.862 samples/sec |         1× |           1×
pipeline-1 |  81.796 samples/sec |     0.853× |       0.800×
pipeline-2 | 135.539 samples/sec |     1.414× |       1.418×
pipeline-4 | 265.958 samples/sec |     2.774× |       2.182×
pipeline-8 | 411.662 samples/sec |     4.294× |       2.891×

(The speed up from Huang et al. is estimated from the figure.)

## Optimized Environment

- torchgpipe 0.0.5
- Python 3.6.9
- PyTorch 1.3.0
- CUDA 10.1.243
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
$ python main.py baseline
$ python main.py pipeline-1
$ python main.py pipeline-2
$ python main.py pipeline-4
$ python main.py pipeline-8
```
