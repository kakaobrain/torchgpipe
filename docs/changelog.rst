Changelog
=========

v0.0.2 (WIP)
~~~~~~

Not released yet.

- Added support for PyTorch 1.1.
- Refined public APIs.
- Proper exceptions for invalid usage.
- Provided inspecting utilities: :func:`torchgpipe.current_microbatch` and
  :func:`torchgpipe.is_recomputing`
- Reimplemented deferred batch normalization by subclassing.
- Provided :mod:`torchgpipe_balancing` for automatic balancing.

v0.0.1
~~~~~~

Released on May 14, 2019 to evaluate usability and efficiency internally.

- Provided a functional GPipe implementation, including pipeline parallelism,
  checkpointing, and deferred batch normalization.
- Supported Python 3.6+ and PyTorch 1.0.
