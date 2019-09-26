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
Your module must be :class:`nn.Sequential <torch.nn.Sequential>` as GPipe will
automatically split the module into partitions with consecutive layers.
`balance` argument determines the number of layers in each partition. `chunks`
argument specifies the number of micro-batches. Input, output, and intermediate
tensors must be ``Tensor`` or ``Tuple[Tensor, ...]``. See also `Restrictions`_
for more details.

The below example code shows how to split a module with four layers into four
partitions each having a single layer. This code also splits a mini-batch into
8 micro-batches::

   from torchgpipe import GPipe

   model = nn.Sequential(a, b, c, d)
   model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)

   for input in data_loader:
       output = model(input)

GPipe optimizes training using CUDA. You should not move the module to a GPU
yourself, because GPipe automatically moves each partition over different
devices. By default, available GPUs starting from ``cuda:0`` are selected in
order for each partition. You can also specify GPUs to use with `devices`
parameter::

   model = GPipe(model,
                 balance=[1, 1, 1, 1],
                 devices=[4, 5, 6, 7],  # Specify GPUs.
                 chunks=8)

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
      TypeError: non-sequential module cannot be partitioned

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
       def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
           return layer1(input), input

   class Layer2(nn.Module):
       #         ┌────────────────┐
       # input --│---> layer2 ----│--> output
       #  skip --│----------------│--> skip
       #         └────────────────┘
       def forward(self, input_and_skip: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
           input, skip = input_and_skip
           return layer2(input), skip

   class Layer3(nn.Module):
       #         ┌────────────────┐
       # input --│---> layer3 --+-│--> output
       #  skip --│--------------' │
       #         └────────────────┘
       def forward(self, input_and_skip: Tuple[Tensor, Tensor]) -> Tensor:
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
