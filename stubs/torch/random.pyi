#MODIFIED BY TORCHGPIPE
from contextlib import contextmanager
from typing import Generator, Iterable, Union

from torch import ByteTensor, device


def set_rng_state(new_state: ByteTensor) -> None: ...
def get_rng_state() -> ByteTensor: ...


@contextmanager
def fork_rng(devices: Iterable[Union[device, str, int]] = ..., enabled: bool = ...) -> Generator[None, None, None]: ...
#END
