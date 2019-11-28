# AmoebaNet-D (L, D) Memory Benchmark

This example reproduces memory benchmark on AmoebaNet-D (L, D), as reported in
Table 1 of [GPipe](https://arxiv.org/abs/1811.06965) by Huang et al. The size
of an AmoebaNet-D model is determined by two hyperparameters L and D which are
proportional to the number of layers and filters, respectively.

Every experiment setting is optimized for Tesla P40 GPUs.

## Result

<table>
  <thead>
    <tr>
      <th>Experiment</th>
      <th>baseline</th>
      <th>pipeline-1</th>
      <th>pipeline-2</th>
      <th>pipeline-4</th>
      <th>pipeline-8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AmoebaNet-D (L, D)</th>
      <td>(18, 208)</td>
      <td>(18, 416)</td>
      <td>(18, 544)</td>
      <td>(36, 544)</td>
      <td>(72, 512)</td>
    </tr>
    <tr>
      <th colspan="6">torchgpipe</th>
    </tr>
    <tr>
      <th>Parameters</th>
      <td> 81.5M</td>
      <td>319.0M</td>
      <td>542.7M</td>
      <td> 1.06B</td>
      <td> 1.84B</td>
    </tr>
    <tr>
      <th>Model Memory</th>
      <td> 0.91 GiB</td>
      <td> 3.57 GiB</td>
      <td> 6.07 GiB</td>
      <td>11.80 GiB</td>
      <td>20.62 GiB</td>
    </tr>
    <tr>
      <th>Peak Activation Memory</th>
      <td>Out of memory</td>
      <td>     0.91 GiB</td>
      <td>     3.39 GiB</td>
      <td>     6.91 GiB</td>
      <td>    10.83 GiB</td>
    </tr>
    <tr>
      <th colspan="6">Huang et al.</th>
    </tr>
    <tr>
      <th>Parameters</th>
      <td>  82M</td>
      <td> 318M</td>
      <td> 542M</td>
      <td>1.05B</td>
      <td> 1.8B</td>
    </tr>
    <tr>
      <th>Model Memory</th>
      <td> 1.05GB</td>
      <td>  3.8GB</td>
      <td> 6.45GB</td>
      <td>12.53GB</td>
      <td>24.62GB</td>
    </tr>
    <tr>
      <th>Peak Activation Memory</th>
      <td> 6.26GB</td>
      <td> 3.46GB</td>
      <td> 8.11GB</td>
      <td>15.21GB</td>
      <td>26.24GB</td>
    </tr>
  </tbody>
</table>

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
