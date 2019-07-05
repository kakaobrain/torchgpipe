Changelog
=========

v0.0.3 (WIP)
~~~~~~

Not released yet.

- Fixed hang at a once failed partition.

v0.0.2
~~~~~~

Released on June 26, 2019.

- Added support for PyTorch 1.1.
- Refined public APIs.
- Detailed documentation.
- Proper exceptions for invalid usage.
- Provided :ref:`automatic balancing <Automatic Balancing>`.
- Provided inspecting utilities: :func:`~torchgpipe.current_microbatch` and
  :func:`~torchgpipe.is_recomputing`
- Reimplemented deferred batch normalization by subclassing.

v0.0.1
~~~~~~

Released on May 14, 2019 to evaluate usability and efficiency internally.

- Provided a functional GPipe implementation, including pipeline parallelism,
  checkpointing, and deferred batch normalization.
- Supported Python 3.6+ and PyTorch 1.0.
