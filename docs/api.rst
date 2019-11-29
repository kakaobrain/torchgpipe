API
===

GPipe Module
~~~~~~~~~~~~

.. py:module:: torchgpipe

.. autoclass:: torchgpipe.GPipe(module, balance, \**kwargs)

   .. automethod:: forward(input)

   .. autoattribute:: balance
      :annotation:

   .. autoattribute:: devices
      :annotation:

   .. autoattribute:: chunks
      :annotation:

   .. autoattribute:: checkpoint
      :annotation:

Skip Connections
~~~~~~~~~~~~~~~~

.. py:module:: torchgpipe.skip

.. autodecorator:: torchgpipe.skip.skippable([stash], [pop])

   .. automethod:: torchgpipe.skip.skippable.Skippable.isolate(ns, [only=names])

.. autofunction:: torchgpipe.skip.stash(name, tensor)

.. autofunction:: torchgpipe.skip.pop(name)

.. autoclass:: torchgpipe.skip.Namespace

.. autofunction:: torchgpipe.skip.verify_skippables(module)

Inspecting GPipe Timeline
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torchgpipe.is_checkpointing()

.. autofunction:: torchgpipe.is_recomputing()

.. _torchgpipe.balance:

Automatic Balancing
~~~~~~~~~~~~~~~~~~~

.. py:module:: torchgpipe.balance

.. autofunction:: torchgpipe.balance.balance_by_time(partitions, module, sample, timeout=1.0, device=torch.device('cuda'))

.. autofunction:: torchgpipe.balance.balance_by_size(partitions, module, input, chunks=1, param_scale=2.0, device=torch.device('cuda'))
