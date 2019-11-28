Benchmarks
==========

Every experiment is reproducible on Tesla P40 GPUs. Follow the link to code for
each benchmark.

Transparency
~~~~~~~~~~~~

ResNet-101 Accuracy Benchmark
-----------------------------

==========  ==========  ===============  ============
Batch size  torchgpipe  nn.DataParallel  Goyal et al.
==========  ==========  ===============  ============
256         21.99±0.13       22.02±0.11    22.08±0.06
1K          22.24±0.19       22.04±0.24           N/A
4K          22.13±0.09              N/A           N/A
==========  ==========  ===============  ============

GPipe should be transparent not to introduce additional hyperparameter tuning.
To verify the transparency, we reproduced top-1 error rate of ResNet-101 on
ImageNet, as reported in Table 2(c) of `Accurate, Large Minibatch SGD
<https://arxiv.org/abs/1706.02677>`_ by Goyal et al.

The reproducible code and experiment details are available in
`benchmarks/resnet101-accuracy`_.

.. _benchmarks/resnet101-accuracy:
   https://github.com/kakaobrain/torchgpipe/tree/master/benchmarks/resnet101-accuracy

Memory
~~~~~~

U-Net (B, C) Memory Benchmark
-----------------------------

.. table::
   :widths: 4,4,4,4

   ==========  ============  ==========  ============
   Experiment  U-Net (B, C)  Parameters  Memory usage
   ==========  ============  ==========  ============
   baseline    (6, 72)           362.2M      20.3 GiB
   pipeline-1  (11, 128)          2.21B      20.5 GiB
   pipeline-2  (24, 128)          4.99B      43.4 GiB
   pipeline-4  (24, 160)          7.80B      79.1 GiB
   pipeline-8  (48, 160)         15.82B     154.1 GiB
   ==========  ============  ==========  ============

The table shows how GPipe facilitates scaling U-Net models. `baseline` denotes
the baseline without pipeline parallelism nor checkpointing, and `pipeline-1`,
`-2`, `-4`, `-8` denotes that the model is trained with GPipe with the
corresponding number of partitions.

Here we used a simplified U-Net architecture. The size of a model is determined
by hyperparameters `B` and `C` which are proportional to the number of layers
and filters, respectively.

The reproducible code and experiment details are available in
`benchmarks/unet-memory`_.

.. _benchmarks/unet-memory:
   https://github.com/kakaobrain/torchgpipe/tree/master/benchmarks/unet-memory

AmoebaNet-D (L, D) Memory Benchmark
-----------------------------------

======================  =============  ==========  ==========  ==========  ==========
Experiment              baseline       pipeline-1  pipeline-2  pipeline-4  pipeline-8
======================  =============  ==========  ==========  ==========  ==========
AmoebaNet-D (L, D)          (18, 208)   (18, 416)   (18, 544)   (36, 544)   (72, 512)
**torchgpipe**
-------------------------------------------------------------------------------------
Parameters                      81.5M      319.0M      542.7M       1.06B       1.84B
Model Memory                 0.91 GiB    3.57 GiB    6.07 GiB   11.80 GiB   20.62 GiB
Peak Activation Memory  Out of memory    0.91 GiB    3.39 GiB    6.91 GiB   10.83 GiB
**Huang et al.**
-------------------------------------------------------------------------------------
Parameters                        82M        318M        542M       1.05B        1.8B
Model Memory                   1.05GB       3.8GB      6.45GB     12.53GB     24.62GB
Peak Activation Memory         6.26GB      3.46GB      8.11GB     15.21GB     26.24GB
======================  =============  ==========  ==========  ==========  ==========

The table shows the better memory utilization of AmoebaNet-D with GPipe, as
stated in Table 1 of `GPipe <https://arxiv.org/abs/1811.06965>`_ by Huang et
al. The size of an AmoebaNet-D model is determined by two hyperparameters `L`
and `D` which are proportional to the number of layers and filters,
respectively.

We reproduced the same settings in the paper with regardless of memory capacity
of Tesla P40 GPUs. The reproducible code and experiment details are available
in `benchmarks/amoebanetd-memory`_.

.. _benchmarks/amoebanetd-memory:
   https://github.com/kakaobrain/torchgpipe/tree/master/benchmarks/amoebanetd-memory

Speed
~~~~~

U-Net (5, 64) Speed Benchmark
-----------------------------

==========  ==========  ========
Experiment  Throughput  Speed up
==========  ==========  ========
baseline      28.500/s        1×
pipeline-1    24.456/s    0.858×
pipeline-2    35.502/s    1.246×
pipeline-4    67.042/s    2.352×
pipeline-8    88.497/s    3.105×
==========  ==========  ========

To verify efficiency with skip connections, we measured the throughput of U-Net
with various number of devices. We chose to use U-Net since it has several long
skip connections.

The reproducible code and experiment details are available in
`benchmarks/unet-speed`_.

.. _benchmarks/unet-speed:
   https://github.com/kakaobrain/torchgpipe/tree/master/benchmarks/unet-speed

AmoebaNet-D (18, 256) Speed Benchmark
-------------------------------------

.. table:: (`n`: number of partitions, `m`: number of micro-batches)

   ==========  ==========  ==========  ============
   Experiment  Throughput  torchgpipe  Huang et al.
   ==========  ==========  ==========  ============
   n=2, m=1      26.733/s          1×            1×
   n=2, m=4      41.133/s      1.539×         1.07×
   n=2, m=32     47.386/s      1.773×         1.21×
   n=4, m=1      26.827/s      1.004×         1.13×
   n=4, m=4      44.543/s      1.666×         1.26×
   n=4, m=32     72.412/s      2.709×         1.84×
   n=8, m=1      24.918/s      0.932×         1.38×
   n=8, m=4      70.065/s      2.621×         1.72×
   n=8, m=32    132.413/s      4.953×         3.48×
   ==========  ==========  ==========  ============

The table shows the reproduced speed benchmark on AmoebaNet-D (18, 256), as
reported in Table 2 of `GPipe <https://arxiv.org/abs/1811.06965>`_ by Huang et
al. Note that we replaced `K` in the paper with `n`.

The reproducible code and experiment details are available in
`benchmarks/amoebanetd-speed`_.

.. _benchmarks/amoebanetd-speed:
   https://github.com/kakaobrain/torchgpipe/tree/master/benchmarks/amoebanetd-speed

ResNet-101 Speed Benchmark
--------------------------

==========  ==========  ==========  ============
Experiment  Throughput  torchgpipe  Huang et al.
==========  ==========  ==========  ============
baseline      95.862/s          1×            1×
pipeline-1    81.796/s      0.853×         0.80×
pipeline-2   135.539/s      1.414×         1.42×
pipeline-4   265.958/s      2.774×         2.18×
pipeline-8   411.662/s      4.294×         2.89×
==========  ==========  ==========  ============

The table shows the reproduced speed benchmark on ResNet-101, as reported in
Figure 3(b) of `the fourth version <https://arxiv.org/abs/1811.06965v4>`_ of
GPipe by Huang et al.

The reproducible code and experiment details are available in
`benchmarks/resnet101-speed`_.

.. _benchmarks/resnet101-speed:
   https://github.com/kakaobrain/torchgpipe/tree/master/benchmarks/resnet101-speed
