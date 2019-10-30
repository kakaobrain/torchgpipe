"""Internal utilities."""
from contextlib import contextmanager
from typing import Generator, List

from torch import Tensor, nn

from torchgpipe.balance import blockpartition

__all__: List[str] = []


@contextmanager
def training_sandbox(module: nn.Sequential) -> Generator[None, None, None]:
    """A context manager for training in sandbox mode."""
    training = module.training
    module.train()

    # Clone state to CPU to minimize CUDA memory for sandboxing.
    def clone_to_cpu(v: Tensor) -> Tensor:
        if v.device.type == 'cpu':
            return v.clone()
        return v.cpu()

    state_dict = {k: clone_to_cpu(v) for k, v in module.state_dict().items()}

    yield

    module.load_state_dict(state_dict)
    module.train(training)


def balance_cost(cost: List[int], partitions: int) -> List[int]:
    partitioned = blockpartition.solve(cost, partitions)
    return [len(p) for p in partitioned]
