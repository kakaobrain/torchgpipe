Benchmarks
==========

ResNet-101 Speed Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~

==========  ===================  =======
Experiment  Throughput           Speedup
==========  ===================  =======
naive-1      92.539 samples/sec   1.000x
pipeline-1   69.960 samples/sec   0.756x
pipeline-2  137.788 samples/sec   1.489x
pipeline-4  243.322 samples/sec   2.629x
pipeline-8  404.084 samples/sec   4.367x
==========  ===================  =======

The code is reproducible on Tesla P40 GPUs, and the experiment details
can be found in `examples/resnet101_speed_benchmark`_.

.. _examples/resnet101_speed_benchmark:
   https://github.com/kakaobrain/torchgpipe/tree/master/examples/resnet101_speed_benchmark

ResNet-101 Accuracy Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

================  ===============
Experiment        Top-1 error (%)
================  ===============
dataparallel-256       22.02±0.11
dataparallel-1k        22.04±0.24
pipeline-256           21.99±0.13
pipeline-1k            22.24±0.19
pipeline-4k            22.13±0.09
================  ===============

The code is reproducible on Tesla P40 GPUs, and the experiment details
can be found in `examples/resnet101_accuracy_benchmark`_.

.. _examples/resnet101_accuracy_benchmark:
   https://github.com/kakaobrain/torchgpipe/tree/master/examples/resnet101_accuracy_benchmark

AmoebaNet-D Speed Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~

==========  ===================  =======
Experiment  Throughput           Speedup
==========  ===================  =======
naive-2      14.188 samples/sec   1.000x
pipeline-2   20.346 samples/sec   1.434x
pipeline-4   29.074 samples/sec   2.049x
pipeline-8   34.392 samples/sec   2.424x
==========  ===================  =======

The code is reproducible on Tesla P40 GPUs, and the experiment details
can be found in `examples/amoebanetd_speed_benchmark`_.

.. _examples/amoebanetd_speed_benchmark:
   https://github.com/kakaobrain/torchgpipe/tree/master/examples/amoebanetd_speed_benchmark

AmoebaNet-D Memory Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

==========  ===========  ==========  ================  =================
Experiment  AmoebaNet-D  # of Model  Total Model       Total Peak
            (L, F)       Parameters  Parameter Memory  Activation Memory
==========  ===========  ==========  ================  =================
naive-1     (6, 208)     90M         1.00GB            --
pipeline-1  (6, 416)     358M        4.01GB            6.64GB
pipeline-2  (6, 544)     613M        6.45GB            11.31GB
pipeline-4  (12, 544)    1.16B       13.00GB           18.72GB
pipeline-8  (24, 512)    2.01B       22.42GB           35.78GB
==========  ===========  ==========  ================  =================
