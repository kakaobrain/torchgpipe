Changelog
=========

v0.0.4 (WIP)
~~~~~~

Not released yet.

v0.0.3
~~~~~~

Released on September 30, 2019.

Featured:
   torchgpipe now overlaps copy and computation using the separate CUDA
   streams. Previously, GPU could not compute a partition while copying
   micro-batches across different GPUs because they all happened on the same
   default CUDA stream.

Other Improvements:
   - Added support for PyTorch 1.2.
   - Redesigned the internal pipeline parallelism to represent dependencies
     transparently.
   - Fixed the hanging issue when an exception is raised in a partition.
   - Fixed the unintended size accumulation (`issue #3`_ by `Shiyan Deng`_) of
     :func:`~torchgpipe_balancing.balance_by_size`.

.. _issue #3: https://github.com/kakaobrain/torchgpipe/issues/3
.. _Shiyan Deng: https://github.com/842974287

Breaking Changes:
   - No more support for PyTorch 1.0.
   - Changed type of :attr:`GPipe.devices <torchgpipe.GPipe.devices>` from
     ``tuple`` to ``list``.
   - Removed ``current_microbatch``. This approach turned out to be
     incompatible with checkpointing.

v0.0.2
~~~~~~

Released on June 26, 2019.

- Added support for PyTorch 1.1.
- Refined public APIs.
- Detailed documentation.
- Proper exceptions for invalid usage.
- Provided :ref:`automatic balancing <Automatic Balancing>`.
- Provided inspecting utilities: ``current_microbatch`` (DO NOT USE, deprecated
  since v0.0.3) and :func:`~torchgpipe.is_recomputing`
- Reimplemented deferred batch normalization by subclassing.

v0.0.1
~~~~~~

Released on May 14, 2019 to evaluate usability and efficiency internally.

- Provided a functional GPipe implementation, including pipeline parallelism,
  checkpointing, and deferred batch normalization.
- Supported Python 3.6+ and PyTorch 1.0.
