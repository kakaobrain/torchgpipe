"""Supports efficiency with skip connections."""
from torchgpipe.skip.namespace import Namespace
from torchgpipe.skip.skippable import pop, skippable, stash, verify_skippables

__all__ = ['skippable', 'stash', 'pop', 'verify_skippables', 'Namespace']
