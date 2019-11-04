# AmoebaNet-D Speed Benchmark

This example reproduces speed benchmark on AmoebaNet-D, as reported in Figure
3(a) of the paper. But there is some difference between torchgpipe and GPipe.
We believe that this difference is not caused by the difference of torchgpipe
and GPipe, rather by reimplementing the AmoebaNet-D model in TensorFlow for
PyTorch. Results will be updated whenever a stable and reproducible AmoebaNet-D
in PyTorch is available.

The benchmark cares of only training speed rather than the model's accuracy.
The batch size is adjusted to achieve higher throughput without any large batch
training tricks. This example also doesn't feed actual dataset like ImageNet or
CIFAR-100. Instead, a fake dataset with 50k 3×224×224 tensors is used to
eliminate data loading overhead.

Every experiment setting is optimized for Tesla P40 GPUs.

## Result

Experiment | Throughput         | Speed up
---------- | -----------------: | -------:
naive-2    | 13.884 samples/sec |   1.000x
pipeline-2 | 20.334 samples/sec |   1.465x
pipeline-4 | 30.893 samples/sec |   2.225x
pipeline-8 | 42.263 samples/sec |   3.044x

## Optimized Environment

- torchgpipe 0.0.4
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
$ python main.py naive-2
$ python main.py pipeline-2
$ python main.py pipeline-4
$ python main.py pipeline-8
```
