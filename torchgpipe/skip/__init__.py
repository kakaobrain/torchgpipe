"""Supports efficiency with skip connections."""
from torchgpipe.skip.analysis import verify_skippables
from torchgpipe.skip.namespace import Namespace
from torchgpipe.skip.skippable import pop, skippable, stash

__all__ = ['skippable', 'stash', 'pop', 'verify_skippables', 'Namespace']
