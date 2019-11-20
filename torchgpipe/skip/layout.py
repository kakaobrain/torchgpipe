"""Static skip connection layout of a ``@skippable`` module."""
from typing import Dict, Iterable, List, Tuple

from torchgpipe.skip.namespace import Namespace

__all__: List[str] = []


class SkipLayout:
    """Represents a skip connection layout across partitions."""

    # Skip routes indexed by 'ns, name': {(ns, name): (prev_j, next_j), ...}
    by_ns_name: Dict[Tuple[Namespace, str], Tuple[int, int]]

    # Skip routes indexed by partition number 'j': [[next_j]: [(prev_j, ns, name), ...], ...]
    by_partition: List[List[Tuple[int, Namespace, str]]]

    def __init__(self,
                 num_partitions: int,
                 skip_routes: Dict[Tuple[Namespace, str], Tuple[int, int]],
                 ) -> None:
        # The skip routes are already indexed by 'ns, name'.
        self.by_ns_name = skip_routes

        # Index skip routes by partition number 'j'.
        self.by_partition = [[] for _ in range(num_partitions)]

        for (ns, name), (prev_j, next_j) in skip_routes.items():
            self.by_partition[next_j].append((prev_j, ns, name))

        for p in self.by_partition:
            p.sort()

    def copy_policy(self, next_j: int) -> Iterable[Tuple[int, Namespace, str]]:
        """Generates skip routes for the given destination partition number.
        The skip routes are sorted by source partition number in ascending
        order.

        Yields:
            Each tuple of (source partition number, namespace, name).

        """
        for prev_j, ns, name in self.by_partition[next_j]:
            if prev_j == next_j:
                # This skip tensor will be popped at the same partition where
                # stashed it. In this case, copy is not required.
                continue

            yield (prev_j, ns, name)

    def requires_copy(self, ns: Namespace, name: str) -> bool:
        """Whether the given namespace and name requires partition-to-partition
        copy or not.
        """
        prev_j, next_j = self.by_ns_name.get((ns, name), (-1, -1))
        return prev_j != next_j
