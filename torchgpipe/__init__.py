"""A GPipe implementation in PyTorch."""
from torchgpipe.__version__ import __version__  # noqa
from torchgpipe.checkpoint import is_recomputing
from torchgpipe.gpipe import GPipe

__all__ = ['GPipe', 'is_recomputing']
