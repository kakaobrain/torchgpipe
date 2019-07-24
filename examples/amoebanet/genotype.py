from typing import List, NamedTuple, Tuple

__all__ = ['amoebanetd_genotype']


class Genotype(NamedTuple):
    normal: List[Tuple[str, int]]
    normal_concat: List[int]
    reduce: List[Tuple[str, int]]
    reduce_concat: List[int]


# The AmoebaNet-D genotype is based on the 'Regularized Evolution for Image Classifier
# Architecture Search' paper (https://arxiv.org/pdf/1802.01548.pdf).
amoebanetd_genotype = Genotype(
    normal=[
        ('max_pool_3x3', 0),
        ('conv_1x1____', 0),
        ('skip_connect', 2),
        ('max_pool_3x3', 2),
        ('skip_connect', 0),
        ('conv_7x1_1x7', 1),
        ('conv_1x1____', 1),
        ('conv_7x1_1x7', 1),
        ('avg_pool_3x3', 0),
        ('conv_1x1____', 3),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('max_pool_2x2', 1),
        ('max_pool_3x3', 1),
        ('conv_3x3____', 0),
        ('skip_connect', 2),
        ('conv_7x1_1x7', 2),
        ('max_pool_3x3', 2),
        ('avg_pool_3x3', 2),
        ('conv_1x1____', 3),
        ('skip_connect', 3),
        ('max_pool_2x2', 0),
    ],
    reduce_concat=[4, 5, 6]
)
