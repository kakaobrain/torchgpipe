"""Internal utilities."""
from typing import Iterable, List

from torchgpipe_balancing import blockpartition

__all__: List[str] = []


def balance_cost(cost: Iterable[float], partitions: int) -> List[int]:
    partitioned = blockpartition.solve(list(cost), partitions)
    return [len(p) for p in partitioned]
