from typing import List

from amoebanet.operations import (avg_pool_3x3, conv_1x1, conv_1x7_7x1, conv_3x3, max_pool_2x2,
                                  max_pool_3x3, none)

__all__: List[str] = []

# The genotype for AmoebaNet-D
NORMAL_OPERATIONS = [
    # (0, max_pool_3x3),
    # (0, conv_1x1),
    # (2, none),
    # (2, max_pool_3x3),
    # (0, none),
    # (1, conv_1x7_7x1),
    # (1, conv_1x1),
    # (1, conv_1x7_7x1),
    # (0, avg_pool_3x3),
    # (3, conv_1x1),

    (1, conv_1x1),
    (1, max_pool_3x3),
    (1, none),
    (0, conv_1x7_7x1),
    (0, conv_1x1),
    (0, conv_1x7_7x1),
    (2, max_pool_3x3),
    (2, none),
    (1, avg_pool_3x3),
    (5, conv_1x1),
]

# According to the paper for AmoebaNet-D, 'normal_concat' should be [4, 5, 6]
# just like 'reduce_concat'. But 'normal_concat' in the reference AmoebaNet-D
# implementation by TensorFlow is defined as [0, 3, 4, 6], which is different
# with the paper.
#
# For now, we couldn't be sure which is correct. But the GPipe paper seems to
# rely on the setting of TensorFlow's implementation. With this, we can
# reproduce the size of model parameters reported at Table 1 in the paper,
# exactly.
#
# Regularized Evolution for Image Classifier Architecture Search
#   https://arxiv.org/pdf/1802.01548.pdf
#
# GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism
#   https://arxiv.org/pdf/1811.06965.pdf
#
# The AmoebaNet-D implementation by TensorFlow
#   https://github.com/tensorflow/tpu/blob/c753c0a/models/official/amoeba_net
#
NORMAL_CONCAT = [0, 3, 4, 6]

REDUCTION_OPERATIONS = [
    (0, max_pool_2x2),
    (0, max_pool_3x3),
    (2, none),
    (1, conv_3x3),
    (2, conv_1x7_7x1),
    (2, max_pool_3x3),
    (3, none),
    (1, max_pool_2x2),
    (2, avg_pool_3x3),
    (3, conv_1x1),
]
REDUCTION_CONCAT = [4, 5, 6]
