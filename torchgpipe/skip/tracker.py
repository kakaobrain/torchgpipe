"""Tracks skip tensors on a thread."""
from contextlib import contextmanager
import threading
from typing import Callable, Dict, Generator, List, Optional, Tuple
import weakref

from torch import Tensor

from torchgpipe.checkpoint import is_checkpointing
from torchgpipe.dependency import fork, join
from torchgpipe.microbatch import Batch
from torchgpipe.skip.layout import SkipLayout
from torchgpipe.skip.namespace import Namespace
from torchgpipe.skip.portal import Portal
from torchgpipe.stream import AbstractStream

__all__: List[str] = []


class SkipTracker:
    """Tracks saved skip tensors.

    It will update the given micro-batch in place. This is because when it
    manipulates the underlying skip tensors, the current micro-batch also has
    to be connected with the skip tensors.

    One thread has one skip tracker. Call :func:`current_skip_tracker` to get
    the skip tracker on the current thread.

    """

    def __init__(self) -> None:
        self.tensors: Dict[Tuple[Namespace, str], Optional[Tensor]] = {}

    def save(self, batch: Batch, ns: Namespace, name: str, tensor: Optional[Tensor]) -> None:
        self.tensors[(ns, name)] = tensor

    def load(self, batch: Batch, ns: Namespace, name: str) -> Optional[Tensor]:
        return self.tensors.pop((ns, name))

    def copy(self,
             batch: Batch,
             prev_stream: AbstractStream,
             next_stream: AbstractStream,
             ns: Namespace,
             name: str,
             ) -> None:
        raise TypeError('copy is not supported for non-portal skip tensors')


class SkipTrackerThroughPotals(SkipTracker):
    """Tracks saved skip tensors through portals. The skip tensors will be
    hided into portals so that the autograd engine cannot track them.

    This tracker is only used when the training or evaluating module is wrapped
    with :class:`torchgpipe.GPipe`.

    """

    def __init__(self, skip_layout: SkipLayout) -> None:
        super().__init__()
        self.skip_layout = skip_layout
        self.portals: Dict[Tuple[Namespace, str], Portal] = {}

    def save(self, batch: Batch, ns: Namespace, name: str, tensor: Optional[Tensor]) -> None:
        """Saves a stashed skip tensor into a portal. The skip tensor will be
        connected onto the given micro-batch with :class:`Join`.
        """
        if not self.skip_layout.requires_copy(ns, name):
            super().save(batch, ns, name, tensor)
            return

        if (ns, name) in self.portals:
            # Reuse the existing portal.
            portal = self.portals[(ns, name)]
            portal.keep_tensor(tensor)

        else:
            # Under checkpointing, the tensor should be kept from the first
            # PortalOrange. It will be reused by the second (recomputed)
            # PortalOrange.
            if is_checkpointing():
                tensor_life = 2
            else:
                tensor_life = 1

            portal = Portal(tensor, tensor_life=tensor_life)
            self.portals[(ns, name)] = portal

        phony = portal.blue()
        batch[0] = join(batch[0], phony)

        # If the grad mode is enabled, delete on backpropagation.
        if phony.requires_grad:
            phony.register_hook(self._hook_to_delete(ns, name))

    # TODO: just keep on empty portals.
    def _hook_to_delete(self, ns: Namespace, name: str) -> Callable[[Tensor], Tensor]:
        ref = weakref.ref(self)

        def hook(grad: Tensor) -> Tensor:
            self = ref()
            if self is not None:
                del self.portals[(ns, name)]
            return grad

        return hook

    def load(self, batch: Batch, ns: Namespace, name: str) -> Optional[Tensor]:
        """Loads a skip tensor to pop. The given micro-batch will be connected
        onto the skip tensor with :class:`Fork`. It will return ``None`` if
        there's no such skip tensor.
        """
        if not self.skip_layout.requires_copy(ns, name):
            tensor = super().load(batch, ns, name)
            return tensor

        portal = self.portals[(ns, name)]
        batch[0], phony = fork(batch[0])
        tensor = portal.orange(phony)

        # If the grad mode is disabled, delete on forward propagation.
        if not tensor.requires_grad and not is_checkpointing():
            del self.portals[(ns, name)]

        return tensor

    def copy(self,
             batch: Batch,
             prev_stream: AbstractStream,
             next_stream: AbstractStream,
             ns: Namespace,
             name: str,
             ) -> None:
        """Copies a skip tensor. The given micro-batch and the skip tensor will
        be tied with :class:`Fork` and :class:`Join`.
        """
        assert self.skip_layout.requires_copy(ns, name)

        batch[0], phony = fork(batch[0])

        portal = self.portals[(ns, name)]
        phony = portal.copy(prev_stream, next_stream, phony)

        batch[0] = join(batch[0], phony)


# TODO: rename to thread_local
class State(threading.local):
    def __init__(self) -> None:
        self.skip_tracker: Optional[SkipTracker] = None


_state = State()


@contextmanager
def use_skip_tracker(skip_tracker: SkipTracker) -> Generator[None, None, None]:
    """Registers the given skip tracker on the current thread within a
    context::

        with use_skip_tracker(my_skip_tracker):
            ...

    """
    orig = _state.skip_tracker

    _state.skip_tracker = skip_tracker

    try:
        yield
    finally:
        _state.skip_tracker = orig


def current_skip_tracker() -> SkipTracker:
    """Gets the skip tracker on the current thread."""
    skip_tracker = _state.skip_tracker

    if skip_tracker is None:
        skip_tracker = SkipTracker()
        _state.skip_tracker = skip_tracker

    return skip_tracker
