# U-Net Memory Benchmark

The table shows how GPipe facilitates scaling U-Net models. *baseline* denotes
the baseline without pipeline parallelism nor checkpointing, and *pipeline-1*,
*-2*, *-4*, *-8* denotes that the model is trained with GPipe with the
corresponding number of partitions.

Here we used a simplified U-Net architecture. The size of a model is determined
by hyperparameters B and C which are proportional to the number of layers and
filters, respectively.

Every experiment setting is optimized for Tesla P40 GPUs.

## Result

Experiment | U-Net (B, C) | Parameters | Memory usage
---------- | ------------ | ---------: | -----------:
baseline   | (6, 72)      |     362.2M |     20.3 GiB
pipeline-1 | (11, 128)    |      2.21B |     20.5 GiB
pipeline-2 | (24, 128)    |      4.99B |     43.4 GiB
pipeline-4 | (24, 160)    |      7.80B |     79.1 GiB
pipeline-8 | (48, 160)    |     15.82B |    154.1 GiB

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
