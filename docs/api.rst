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

.. autofunction:: torchgpipe.current_microbatch()

.. autofunction:: torchgpipe.is_recomputing()

Automatic Balancing
~~~~~~~~~~~~~~~~~~~

.. autofunction:: torchgpipe_balancing.balance_by_time(module, canary, partitions, device, timeout)

.. autofunction:: torchgpipe_balancing.balance_by_size(module, canary, partitions, device)
