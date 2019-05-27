"""A GPipe implementation in PyTorch."""
from torchgpipe.__version__ import __version__  # noqa
from torchgpipe.gpipe import GPipe, current_microbatch

__all__ = ['GPipe', 'current_microbatch']
