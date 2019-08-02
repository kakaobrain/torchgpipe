# AmoebaNet-D Performance Benchmark

This example reproduces performance benchmark on AmoebaNet-D, as reported in
Figure 3(a) of the paper. But there is some difference between torchgpipe and
GPipe. We believe that this difference is not caused by the difference of
torchgpipe and GPipe, rather by reimplementing the AmoebaNet-D model in
TensorFlow for PyTorch. Results will be updated whenever a stable and
reproducible AmoebaNet-D in PyTorch is available.

The benchmark cares of only training performance rather than the model's
accuracy. The batch size is adjusted to achieve higher throughput without any
large batch training tricks. This example also doesn't feed actual dataset like
ImageNet or CIFAR-100. Instead, a fake dataset with 50k 3×224×224 tensors is
used to eliminate data loading overhead.

Every experiment setting is optimized for Tesla P40 GPUs.

## Result

Experiment | Throughput        | Speed up
---------- | ----------------: | -------:
naive-2    | 14.18 samples/sec |   1.000x
pipeline-2 | 20.34 samples/sec |   1.434x
pipeline-4 | 29.07 samples/sec |   2.049x
pipeline-8 | 34.39 samples/sec |   2.424x

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
$ python main.py naive-2
$ python main.py pipeline-2
$ python main.py pipeline-4
$ python main.py pipeline-8
```
