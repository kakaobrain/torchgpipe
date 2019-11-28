# Timeline Benchmark with U-Net

Performance of torchgpipe when optimization components are incrementally added.
The U-Net model with (B, C) = (5, 64) is used for the experiment.

Every experiment setting is optimized for Tesla P40 GPUs.

## Result

Experiment  | Optimization components        | Throughput | Utilization | Memory usage
----------- | ------------------------------ | ---------: | ----------: | -----------:
baseline    |                                |   30.662/s |         44% |     52.2 GiB
dep-x-x     | Dependency                     |   41.306/s |         59% |     19.1 GiB
dep-str-x   | Dependency + Streams           |   55.191/s |         71% |     30.0 GiB
dep-str-ptl | Dependency + Streams + Portals |   58.477/s |         75% |     23.5 GiB

## Optimized Environment

- torchgpipe 0.0.5
- Python 3.6.9
- PyTorch 1.3.0
- CUDA 10.1.243
- 4 Tesla P40 GPUs
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
$ python main.py dep-x-x
$ python main.py dep-str-x
$ python main.py dep-str-ptl
```
