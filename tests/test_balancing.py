import pytest

from torchgpipe_balancing import blockpartition


def test_blockpartition():
    assert blockpartition.solve([1, 2, 3, 4, 5, 6], partitions=2) == [[1, 2, 3, 4], [5, 6]]


def test_blockpartition_zeros():
    assert blockpartition.solve([0, 0], partitions=2) == [[0], [0]]


def test_blockpartition_non_positive_partitions():
    with pytest.raises(ValueError):
        blockpartition.solve([42], partitions=0)
    with pytest.raises(ValueError):
        blockpartition.solve([42], partitions=-1)


def test_blockpartition_short_sequence():
    with pytest.raises(ValueError):
        blockpartition.solve([], partitions=1)
    with pytest.raises(ValueError):
        blockpartition.solve([42], partitions=2)
