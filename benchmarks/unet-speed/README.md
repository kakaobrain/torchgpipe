# U-Net Speed Benchmark

To verify efficiency with skip connections, we measured the throughput of U-Net
with various number of devices. We chose to use U-Net since it has several long
skip connections.

Every experiment setting is optimized for Tesla P40 GPUs.

## Result

Experiment | Throughput | Speed up
---------- | ---------: | -------:
baseline   |   28.500/s |       1×
pipeline-1 |   24.456/s |   0.858×
pipeline-2 |   35.502/s |   1.246×
pipeline-4 |   67.042/s |   2.352×
pipeline-8 |   88.497/s |   3.105×

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
