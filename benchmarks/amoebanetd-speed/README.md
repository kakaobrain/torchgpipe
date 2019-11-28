# AmoebaNet-D (18, 256) Speed Benchmark

The table shows the reproduced speed benchmark on AmoebaNet-D (18, 256), as
reported in Table 2 of [GPipe](https://arxiv.org/abs/1811.06965) by Huang et
al. Note that we replaced *K* in the paper with *n*.

The benchmark cares of only training speed rather than the model's accuracy.
The batch size is adjusted to achieve higher throughput without any large batch
training tricks. This example also doesn't feed the actual ImageNet dataset.
Instead, fake 3×224×224 tensors over 1000 labels are used to eliminate data
loading overhead.

Every experiment setting is optimized for Tesla P40 GPUs.

## Result

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
$ python main.py n2m1
$ python main.py n2m4
$ python main.py n2m32
$ python main.py n4m1
$ python main.py n4m4
$ python main.py n4m32
$ python main.py n8m1
$ python main.py n8m4
$ python main.py n8m32
```
