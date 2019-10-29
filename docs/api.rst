API
===

GPipe Module
~~~~~~~~~~~~

.. autoclass:: torchgpipe.GPipe(module, balance, \**kwargs)

   .. automethod:: forward(input)

   .. autoattribute:: devices
      :annotation:

Inspecting GPipe Timeline
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torchgpipe.is_recomputing()

Automatic Balancing
~~~~~~~~~~~~~~~~~~~

.. autofunction:: torchgpipe.balance.balance_by_time(partitions, module, sample, timeout=1.0)

.. autofunction:: torchgpipe.balance.balance_by_size(partitions, module, input, chunks=1, param_scale=2.0)
