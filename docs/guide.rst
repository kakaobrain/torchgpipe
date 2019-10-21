User Guide
==========

Installation
~~~~~~~~~~~~

:mod:`torchgpipe` is available on PyPI_. Install by ``pip``:

.. sourcecode:: console

   $ pip install torchgpipe

.. _PyPI: https://pypi.org/project/torchgpipe

Python 3.6+ (CPython) is required.

PyTorch 1.1+ will be installed automatically if you don't have a satisfied one.
However, we highly recommend you to use the latest version of PyTorch.

Applying GPipe
~~~~~~~~~~~~~~

To train a module with GPipe, simply wrap it with :class:`torchgpipe.GPipe`.
Your module must be a :class:`nn.Sequential <torch.nn.Sequential>` as GPipe
will automatically split the module into partitions. A partition is a group of
consecutive layers that run on a single device together. `balance` argument
determines the number of layers in each partition. `chunks` argument specifies
the number of micro-batches. Input, output, and intermediate tensors must be
``Tensor`` or ``Tuple[Tensor, ...]``. See also `Restrictions`_ for more
details.

The below example code shows how to split a module with four layers into two
partitions each having two layers. This code also splits a mini-batch into 8
micro-batches::

   from torchgpipe import GPipe

   model = nn.Sequential(a, b, c, d)
   model = GPipe(model, balance=[2, 2], chunks=8)

   # 1st partition: nn.Sequential(a, b) on cuda:0
   # 2nd partition: nn.Sequential(c, d) on cuda:1

   for input in data_loader:
       output = model(input)

GPipe optimizes training using CUDA. You should not move the module to a GPU
yourself, because GPipe automatically moves each partition over different
devices. By default, available GPUs starting from ``cuda:0`` are selected in
order for each partition. You can also specify GPUs to use with `devices`
parameter::

   model = GPipe(model,
                 balance=[2, 2],
                 devices=[4, 2],  # Specify GPUs.
                 chunks=8)

The typical model parallelism is a special case of GPipe. GPipe without
micro-batches and checkpointing is equivalent to model parallelism. You can
disable them with ``chunks=1`` and ``checkpoint='never'`` options::

   model = GPipe(model, balance=[2, 2], chunks=1, checkpoint='never')

Input and Output Device
~~~~~~~~~~~~~~~~~~~~~~~

Unlike a typical module, with GPipe, the input device is different from the
output device except for when there is only one partition. This is because the
first partition and last partition are placed in different devices.

Therefore, you have to move the input and target to the corresponding devices.
It can be done with :attr:`GPipe.devices <torchgpipe.GPipe.devices>`, which
holds the list of devices for each partition::

   in_device = model.devices[0]
   out_device = model.devices[-1]

   for input, target in data_loader:
       # input on in_device
       input = input.to(in_device, non_blocking=True)

       # target on out_device
       target = target.to(out_device, non_blocking=True)

       # output on out_device
       output = model(input)
       loss = F.cross_entropy(output, target)
       loss.backward()
       ...

Automatic Balancing
~~~~~~~~~~~~~~~~~~~

It could be hard to determine the optimal balance of a model. In particular, if
you are still designing a model, the model architecture may change over time.
In this case, we highly recommend :mod:`torchgpipe_balancing` for automatic
balancing. This won't give you the optimal balance, but a good-enough balance.
Note that this is provided by `torchgpipe` package, and is not from the GPipe
paper.

There are two balancing tools, :func:`~torchgpipe_balancing.balance_by_time`
and :func:`~torchgpipe_balancing.balance_by_size`. Both are based on per-layer
profiling. Just like `PyTorch JIT`_, you need to feed a sample input into the
model. :func:`~torchgpipe_balancing.balance_by_time` traces elapsed time of
each layer, while :func:`~torchgpipe_balancing.balance_by_size` detects the
CUDA memory usage of each layer. Choose the balancing tool for your needs::

   from torchgpipe import GPipe
   from torchgpipe_balancing import balance_by_time

   sample = torch.rand(128, 3, 224, 224)
   balance = balance_by_time(model, sample, partitions=4)

   model = GPipe(model, balance, chunks=8)

.. _PyTorch JIT: https://pytorch.org/docs/stable/jit.html

Trade-offs
~~~~~~~~~~

Number of Micro-batches
-----------------------

Number of micro-batches has a trade-off between GPU utilization per micro-batch
and total area of bubble. You need to find the best number of micro-batches for
your model.

GPU may slow down when processing many small micro-batches compared to larger
micro-batches. GPU will not be fully utilized if each CUDA kernel is too cheap
to compute, hence too small micro-batches cause underutilization. On the other
hand, the area of bubble is minimized when the size of each micro-batch is
minimal. Ideally, you should choose the largest number of micro-batches that
doesn't underutilize GPUs.

As a side note, BatchNorm tends to perform worse with smaller batch size. Large
number of micro-batches may affect the final performance of model using
BatchNorm negatively just like in :class:`nn.DataParallel
<torch.nn.DataParallel>`.

Checkpointing
-------------

Checkpointing drastically helps to reduce memory usage, but the overall
training would slow down by about 25%. You can handle how to apply
checkpointing on your model. There are three options:

- ``always`` -- Apply checkpointing over all micro-batches.
- ``except_last`` (default) -- Apply checkpointing except the last micro-batch.
- ``never`` -- Checkpointing is never applied.

Usually, checkpointing at the last micro-batch may not be useful because the
saved memory will be reconstructed immediately. That's why we choose
``except_last`` as the default option.

If you decide not to use checkpointing at all, :class:`nn.DataParallel
<torch.nn.DataParallel>` might be more efficient than GPipe.

Referential Transparency
~~~~~~~~~~~~~~~~~~~~~~~~

Checkpointing executes forward propagation again at backpropagation, which is
called `recomputation`. We assume that both the executions are identical.
Hence, all layers should be `referentially transparent
<https://en.wikipedia.org/wiki/Referential_transparency>`_ in forward
propagation. Here are the typical cases that break referential transparency:

In-place Operations:
   We do not recommend using in-place operations with checkpointing.
   Especially, if an in-place operation such as ``add_(1)`` is applied to the
   input of a checkpointed partition, then the recomputation can't recover the
   original input.

Randomness not managed by PyTorch:
   The randomness managed by PyTorch, including :func:`torch.manual_seed`,
   :func:`torch.rand`, or :class:`nn.Dropout <torch.nn.Dropout>`, is
   deterministically reproduced in recomputation. But other randomnesses, such
   as Python standard :mod:`random` or :mod:`numpy.random`, are not. We highly
   recommend to use PyTorch randomness for referential transparency.

Side Effects:
   Some modules such as BatchNorm update their state in forward propagation.
   Hence, updated state in recomputation might not be identical to the original
   state.

Restrictions
~~~~~~~~~~~~

If you get any errors, check the following restrictions first.

Sequential:
   Your module must be :class:`nn.Sequential <torch.nn.Sequential>`. For
   example, the models in :mod:`torchvision` are not sequential. They can't be
   wrapped by :class:`~torchgpipe.GPipe` directly::

      >>> from torchvision.models.resnet import resnet101
      >>> model = resnet101()
      >>> type(model)
      torchvision.models.resnet.ResNet
      >>> GPipe(model, balance=..., chunks=...)
      Traceback (most recent call last)
        ...
      TypeError: module must be nn.Sequential to be partitioned

   See `the sequential ResNet example`_ to figure out how to make a  model into
   a :class:`nn.Sequential <torch.nn.Sequential>` model.

   .. _the sequential ResNet example:
      https://github.com/kakaobrain/torchgpipe/tree/master/examples/resnet

   :class:`nn.Sequential <torch.nn.Sequential>` assumes that every underlying
   layer takes only one argument. Calling ``forward(x)`` on
   ``nn.Sequential(A(), B(), C())`` is essentially the same as calling
   ``C(B(A(x)))``. Hence, you can't design an underlying layer with multiple
   arguments::

      class MyModule(nn.Module):
          def forward(self, a, b, c):
              return a + b - c

      model = nn.Sequential(..., MyModule(), ...)
      model(input)  # FAILS!

Tensor or Tensors:
   As we discussed above, each layer must take only one argument due to
   :class:`nn.Sequential <torch.nn.Sequential>`. There is one more restriction.
   Every underlying layers' input and output must be ``Tensor`` or
   ``Tuple[Tensor, ...]``::

      # OK
      def forward(input: Tensor) -> Tensor: ...
      def forward(input: Tensor) -> Tuple[Tensor, Tensor]: ...
      def forward(input: Tuple[Tensor, Tensor]) -> Tensor: ...

      # Error
      def forward(input1: Tensor, input2: Tensor) -> Tensor: ...
      def forward(input: Tensor, label: str) -> Tensor: ...
      def forward(input: Tensor) -> Dict[str, Tensor]: ...
      def forward(input: Tensor) -> Tuple[Tensor, str]: ...

   The reason is that GPipe can't assume how the non-tensor inputs for a
   mini-batch can be split for micro-batches.

Unique Parameters:
   :class:`~torchgpipe.GPipe` places each partition on the corresponding
   device. When placing a partition, the parameters of the partition are also
   moved to the destination. GPipe cannot support a module with a parameter on
   two or more devices::

      >>> conv1 = nn.Conv2d(3, 3, 1)
      >>> conv2 = nn.Conv2d(3, 3, 1)
      >>> conv1.weight = conv2.weight
      >>> model = nn.Sequential(conv1, conv2)
      >>> model = GPipe(model, balance=[1, 1], ...)
      Traceback (most recent call last)
        ...
      ValueError: module with duplicate parameters in distinct children is not supported

Complex Modules
~~~~~~~~~~~~~~~

This part of the documentation discusses how to implement a complex module
compatible with :class:`~torchgpipe.GPipe`. First, you should understand how
GPipe works. See :ref:`Understanding GPipe`.

Skip Connections
----------------

Many deep learning models, such as ResNet or AmoebaNet, contain skip
connections. There are two ways to implement skip connections. Let's assume we
have to implement a skip connection like this::

   latent = layer1(input)
   latent = layer2(latent)
   output = layer3(latent) + input  # skip connection

To make this module sequential, we define modules for each layer. Simply,
a skip connection can be implemented by making underlying layers with
``Tuple[Tensor, Tensor]`` parameter and return type::

   class Layer1(nn.Module):
       #         ┌────────────────┐
       # input --│-+-> layer1 ----│--> output
       #         │ '--------------│--> skip
       #         └────────────────┘
       def forward(self, input):
           skip = input
           return layer1(input), skip

   class Layer2(nn.Module):
       #         ┌────────────────┐
       # input --│---> layer2 ----│--> output
       #  skip --│----------------│--> skip
       #         └────────────────┘
       def forward(self, input_and_skip):
           input, skip = input_and_skip
           return layer2(input), skip

   class Layer3(nn.Module):
       #         ┌────────────────┐
       # input --│---> layer3 --+-│--> output
       #  skip --│--------------' │
       #         └────────────────┘
       def forward(self, input_and_skip):
           input, skip = input_and_skip
           return layer3(input) + skip

   model = nn.Sequential(Layer1(), Layer2(), Layer3())

Because of the skip connection being represented as a normal parameter, GPipe
can move the tensors from partition to partition::

   model = GPipe(model, balance=[1, 1, 1], chunks=8)

It is the most straightforward approach to implement skip connections. But
there is a disadvantage. In the above example, the skip tensor is copied to the
second device, but it is never used on the second device. Unnecessarily copying
tensor wastes time and memory.

Detecting Recomputation
-----------------------

Checkpointing in GPipe performs forward propagations twice. The second forward
propagation is called `recomputation`. This may cause a problem when a module
such as :class:`nn.BatchNorm2d <torch.nn.BatchNorm2d>` updates its running
estimates of batch statistics on each forward propagation. It should not update
the running estimates again during the recomputation. To avoid updating the
running estimates twice, modules' ``forward`` method needs be able to detect
that this is the recomputation.

It can be done by :func:`~torchgpipe.is_recomputing`. This function returns
``True`` if called during the recomputation::

   class Counter(nn.Module):
       def __init__(self):
           super().__init__()
           self.counter = 0

       def forward(self, input):
           if not is_recomputing():
               self.counter += 1
           return input

.. note::

   ``deferred_batch_norm=True`` on :class:`~torchgpipe.GPipe` will prevent
   updating the running statistics twice.
