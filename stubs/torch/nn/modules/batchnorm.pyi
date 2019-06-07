#MODIFIED BY TORCHGPIPE
from typing import Any, Callable, Iterable, Iterator, Optional

from torch import Tensor
from torch.nn import Module, Parameter


class _BatchNorm(Module):
    num_features: int
    eps: float
    momentum: Optional[float]
    affine: bool
    track_running_stats: bool

    weight: Parameter
    bias: Parameter
    running_mean: Tensor
    running_var: Tensor
    num_batches_tracked: Tensor

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: Optional[float] = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 ) -> None: ...

#END
