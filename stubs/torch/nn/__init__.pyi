from .modules import *
from .parameter import Parameter as Parameter
from .parallel import DataParallel as DataParallel
from . import init as init
from . import utils as utils

#MODIFIED BY TORCHGPIPE
from .. import Tensor
class Flatten(Module):
    start_dim: int
    end_dim: int
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
#END
