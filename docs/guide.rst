User Guide
==========

Installation
~~~~~~~~~~~~

:mod:`torchgpipe` is available on PyPI_. Install by ``pip``:

.. sourcecode:: console

   $ pip install torchgpipe

.. _PyPI: https://pypi.org/project/torchgpipe

Python 3.6+ (CPython) is required.

PyTorch 1.0+ will be installed automatically if you don't have a satisfied one.
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
order for each partition. You can also specify GPUs to select by `devices`
parameter::

   mode = GPipe(model,
                balance=[1, 1, 1, 1],
                devices=[4, 5, 6, 7],  # Specify GPUs.
                chunks=8)

Input and Output Device
~~~~~~~~~~~~~~~~~~~~~~~

Unlike a typical module, with GPipe, the input device is different from the
output device except there is only one partition. This is because the first
partition and last partition should be placed in different devices.

Therefore, you should move the input and target to the corresponding devices.
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
you are still designing a model, probably the model architecture may change
over time. In this case, we highly recommend :mod:`torchgpipe_balancing` for
automatic balancing. This library is also a part of `torchgpipe` package but
not a part of the GPipe paper.

There are two balancing tools, :func:`~torchgpipe_balancing.balance_by_time`
and :func:`~torchgpipe_balancing.balance_by_size`. Both are based on per-layer
profiling. Just like `PyTorch JIT`_, you need to feed a sample input into the
model. :func:`~torchgpipe_balancing.balance_by_time` traces elapsed time of
each layer, while :func:`~torchgpipe_balancing.balance_by_size` detects the
CUDA memory usage of each layer. Choose a balancing tool for your needs::

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

To make this module sequential, we will define modules for each layer. Simply,
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
there is a disadvantage. In the above example, the skipping input tensor is
copied to the second device, but it is never used at the device. Unnecessarily
copied tensor wastes time and memory.

The following section introduces alternative approach for skip connection.

Long Skip Connections
---------------------

The disadvantage mentioned above might be catastrophic if the unnecessarily
copied tensor is very large, or it is copied over many devices. The second case
often occurs when implementing long skip connections. Let's assume now we have
8 layers between input and output::

   latent = layer1(input)
   latent = layer2(latent)
   latent = layer3(latent)
   latent = layer4(latent)
   latent = layer5(latent)
   latent = layer6(latent)
   latent = layer7(latent)
   output = layer8(latent) + input  # skip connection

With the prior approach, GPipe will copy the skipping input tensor to all
devices, but 6 of them are unnecessary. The alternative approach is managing
the skipping tensor manually in the module code. Now we will introduce a shared
memory between ``Layer1`` and ``Layer8`` to toss the tensor without going
through regardless layers. We might use a global variable named ``skip_buf``
for the shared memory. But actually, this approach doesn't work::

   # !!!!!!!!!!!!!!!!!!!!!!!!!
   # THIS IS A FAILING EXAMPLE
   # !!!!!!!!!!!!!!!!!!!!!!!!!

   # The shared memory between Layer1 and Layer8.
   skip_buf = None

   class Layer1(nn.Module):
       def forward(self, input: Tensor) -> Tensor:
           # Remember the skipping tensor.
           global skip_buf
           skip_buf = input

           return layer1(input)

   class Layer2(nn.Module):
       def forward(self, input: Tensor) -> Tensor:
           return layer2(input)

   ...  # Layer3-7 are similar to Layer2.

   class Layer8(nn.Module):
       def forward(self, input: Tensor) -> Tensor:
           # Retrieve the skipping tensor.
           global skip_buf
           skip = skip_buf

           # Release the shared memory.
           skip_buf = None

           # The tensor should be copied to the device manually.
           skip = skip.to(input.device)

           return layer8(input) + skip

Each layer is executed several times due to the multiple micro-batches.
Partitions work together concurrently, so the shared memory would be
overwritten in non-deterministic order.

For example, when ``Layer8`` processes the first micro-batch, it might receive
the third (or any) micro-batch as a skipping tensor if ``Layer1`` has just
processed the latter micro-batch. We need to separate the shared memory for
different micro-batches.

Therefore, the key is an identifier of each micro-batch. How do we identify
which micro-batch the partition is currently processing? :mod:`torchgpipe`
provides :func:`~torchgpipe.current_microbatch` for this purpose. If you call
the function in a GPipe context, it will return some tensor. The tensor
identifies the current micro-batch. You can use this as simply a dictionary
key, or the target of a weak reference::

   from torchgpipe import current_microbatch

   # The shared memory between Layer1 and Layer8 indexed by micro-batch.
   skips: Dict[Tensor, Tensor] = {}

   class Layer1(nn.Module):
       def forward(self, input: Tensor) -> Tensor:
           # Remember the skipping tensor per micro-batch.
           skips[current_microbatch()] = input

           return layer1(input)

   ...  # Layer2-7 are folded.

   class Layer8(nn.Module):
       def forward(self, input: Tensor) -> Tensor:
           # Retrieve the skipping tensor for the current micro-batch.
           skip = skips.pop(current_microbatch())

           # The tensor should be copied to the device manually.
           skip = skip.to(input.device)

           return layer8(input) + skip

This approach is not required to everyone. Furthermore, we didn't intend to
modify user's module to apply GPipe. However, long skip connections are one of
the common building blocks in modern CNN models. That is why we have provided
this functionality.

Detecting Recomputation
-----------------------

Checkpointing in GPipe performs forward propagations twice. The second
forward propagation is called `recomputation`. This may cause a problem when a
module such as :class:`nn.BatchNorm2d <torch.nn.BatchNorm2d>` updates its
buffers on each forward propagation. It should not update the buffers again
during the recomputation. To achieve it, modules' ``forward`` method should be
able to detect that is the recomputation or the first forward progagation.

It can be done by :func:`~torchgpipe.is_recomputing`. This function returns
``True`` if the code is running on the recomputation::

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
