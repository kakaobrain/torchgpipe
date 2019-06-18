torchgpipe
==========

A GPipe_ implementation in PyTorch_.

.. _GPipe: https://arxiv.org/abs/1811.06965
.. _PyTorch: https://pytorch.org/

.. sourcecode:: python

   from torchgpipe import GPipe

   model = nn.Sequential(a, b, c, d)
   model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)

   for input in data_loader:
       output = model(input)


Installing
----------

torchgpipe is available on PyPI_.

.. sourcecode:: console

   $ pip install torchgpipe

.. _PyPI: https://pypi.org/project/torchgpipe


API
---

.. autoclass:: torchgpipe.GPipe(module, balance, \**kwargs)

   .. automethod:: forward(input)

   .. autoattribute:: devices
      :annotation:

.. autofunction:: torchgpipe_balancing.balance_by_time(module, canary, partitions, device, timeout)

.. autofunction:: torchgpipe_balancing.balance_by_size(module, canary, partitions, device)

Licensing and Authors
---------------------

This package is opened under the Apache License 2.0.

We are `Heungsub Lee`_ and Myungryong Jeong in `Kakao Brain`_.

.. _Heungsub Lee: https://subl.ee/
.. _Kakao Brain: https://kakaobrain.com/
