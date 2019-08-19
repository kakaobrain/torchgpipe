Benchmarks
==========

ResNet-101
~~~~~~~~~~

ResNet-101 Performance Benchmark
--------------------------------

==========  ===================  =======
Experiment  Throughput           Speedup
==========  ===================  =======
naive-1     100.922 samples/sec   1.000x
pipeline-1   74.128 samples/sec   0.735x
pipeline-2  136.929 samples/sec   1.357x
pipeline-4  238.058 samples/sec   2.359x
pipeline-8  328.563 samples/sec   3.256x
==========  ===================  =======

The code is reproducible on Tesla P40 GPUs, and the experiment details
can be found in `examples/resnet101_performance_benchmark`_.

.. _examples/resnet101_performance_benchmark:
   https://github.com/kakaobrain/torchgpipe/tree/master/examples/resnet101_performance_benchmark

AmoebaNet-D
~~~~~~~~~~~

AmoebaNet-D Performance Benchmark
---------------------------------

==========  ===================  =======
Experiment  Throughput           Speedup
==========  ===================  =======
naive-2      14.188 samples/sec   1.000x
pipeline-2   20.346 samples/sec   1.434x
pipeline-4   29.074 samples/sec   2.049x
pipeline-8   34.392 samples/sec   2.424x
==========  ===================  =======

The code is reproducible on Tesla P40 GPUs, and the experiment details
can be found in `examples/amoebanetd_performance_benchmark`_.

.. _examples/amoebanetd_performance_benchmark:
   https://github.com/kakaobrain/torchgpipe/tree/master/examples/amoebanetd_performance_benchmark

AmoebaNet-D Memory Benchmark
----------------------------

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
