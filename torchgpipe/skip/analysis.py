"""Analyzes a module including ``@skippable`` statically ahead of time."""
from typing import Dict, List, Set, Tuple

from torch import nn

from torchgpipe.skip.layout import SkipLayout
from torchgpipe.skip.namespace import Namespace
from torchgpipe.skip.skippable import Skippable

__all__ = ['verify_skippables']


def verify_skippables(module: nn.Sequential) -> None:
    """Verifies if the underlying skippable modules satisfy integrity.

    Every skip tensors must have only one pair of ``stash`` and ``pop``. If
    there are one or more unmatched pairs, it will raise :exc:`TypeError` with
    detailed messages.

    Raises:
        TypeError:
            one or more pairs of ``stash`` and ``pop`` are not matched.

    """
    stashed: Set[Tuple[Namespace, str]] = set()
    popped: Set[Tuple[Namespace, str]] = set()
    msgs: List[str] = []

    for layer_name, layer in module.named_children():
        if not isinstance(layer, Skippable):
            continue

        for name in layer.stashable_names & layer.poppable_names:
            msg = "'%s' declared '%s' both as stashable and as poppable" % (layer_name, name)
            msgs.append(msg)

        for ns, name in layer.stashable():
            if name in layer.poppable_names:
                continue

            if (ns, name) in stashed:
                msg = "'%s' redeclared '%s' as stashable" % (layer_name, name)
                msgs.append(msg)
                continue

            stashed.add((ns, name))

        for ns, name in layer.poppable():
            if name in layer.stashable_names:
                continue

            if (ns, name) in popped:
                msg = "'%s' redeclared '%s' as poppable" % (layer_name, name)
                msgs.append(msg)
                continue

            if (ns, name) not in stashed:
                msg = "'%s' declared '%s' as poppable but it was not stashed" \
                      '' % (layer_name, name)
                msgs.append(msg)
                continue

            popped.add((ns, name))

    for (_, name) in stashed - popped:
        msg = "any module did not declare '%s' as poppable" % (name,)
        msgs.append(msg)

    if msgs:
        raise TypeError('one or more pairs of stash and pop not matched:\n\n%s'
                        '' % '\n'.join('* %s' % x for x in msgs))


def inspect_skip_layout(partitions: List[nn.Sequential]) -> SkipLayout:
    """Inspects the skip connection layout in the given partitions."""
    skip_routes: Dict[Tuple[Namespace, str], Tuple[int, int]] = {}
    stashed_at: Dict[Tuple[Namespace, str], int] = {}

    for j, partition in enumerate(partitions):
        for layer in partition:
            if not isinstance(layer, Skippable):
                continue

            for ns, name in layer.stashable():
                stashed_at[(ns, name)] = j

            for ns, name in layer.poppable():
                prev_j = stashed_at.pop((ns, name))
                skip_routes[(ns, name)] = (prev_j, j)

    return SkipLayout(len(partitions), skip_routes)
