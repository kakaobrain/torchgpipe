Changelog
=========

v0.0.3 (WIP)
~~~~~~

Not released yet.

- Added support for PyTorch 1.2.
- Redesigned the internal pipeline parallelism for bidirectional deterministic
  lockstep.
- Optimized by using separate CUDA streams for device-to-device copies.
- Fixed hang at a once failed partition.
- Removed ``current_microbatch`` which actually didn't work.
- Fixed the size cumulation (`issue #3`_ by `Shiyan Deng`_) of
  :func:`~torchgpipe_balancing.balance_by_size`.

.. _issue #3: https://github.com/kakaobrain/torchgpipe/issues/3
.. _Shiyan Deng: https://github.com/842974287

v0.0.2
~~~~~~

Released on June 26, 2019.

- Added support for PyTorch 1.1.
- Refined public APIs.
- Detailed documentation.
- Proper exceptions for invalid usage.
- Provided :ref:`automatic balancing <Automatic Balancing>`.
- Provided inspecting utilities: ``current_microbatch`` (deprecated since
  v0.0.3) and :func:`~torchgpipe.is_recomputing`
- Reimplemented deferred batch normalization by subclassing.

v0.0.1
~~~~~~

Released on May 14, 2019 to evaluate usability and efficiency internally.

- Provided a functional GPipe implementation, including pipeline parallelism,
  checkpointing, and deferred batch normalization.
- Supported Python 3.6+ and PyTorch 1.0.
