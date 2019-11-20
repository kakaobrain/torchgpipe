"""Provides isolated namespace of skip tensors."""
import abc
from functools import total_ordering
from typing import Any
import uuid

__all__ = ['Namespace']


@total_ordering
class Namespace(metaclass=abc.ABCMeta):
    """An isolated namespace for skip tensors.

    Skip tensors having the same name can be declared in a single sequential
    module if they are isolated by different namespaces.

    Usage::

        @skippable(stash=['skip'])
        class Stash(nn.Module):
            def forward(self, x):
                yield stash('skip', x)
                return x + 1

        @skippable(pop=['skip'])
        class Pop(nn.Module):
            def forward(self, x):
                skip = yield pop('skip')
                return x + skip + 1

        ns1 = Namespace()
        ns2 = Namespace()

        model = nn.Sequential(
            Stash().isolate(ns1),
            Stash().isolate(ns2),
            Pop().isolate(ns2),
            Pop().isolate(ns1),
        )

    """
    __slots__ = ('id',)

    def __init__(self) -> None:
        self.id = uuid.uuid4()

    def __repr__(self) -> str:
        return f"<Namespace '{self.id}'>"

    def __hash__(self) -> int:
        return hash(self.id)

    # Namespaces should support ordering, since SkipLayout will sort tuples
    # including a namespace. But actual order between namespaces is not
    # necessary. That's why they are ordered by version 4 UUID which generates
    # random numbers.
    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Namespace):
            return self.id < other.id
        return False

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Namespace):
            return self.id == other.id
        return False


# 'None' is the default namespace,
# which means that 'isinstance(None, Namespace)' is 'True'.
Namespace.register(type(None))
