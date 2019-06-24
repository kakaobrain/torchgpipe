#MODIFIED BY TORCHGPIPE
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union, overload

from torch import Tensor, device


TModule = TypeVar('TModule', bound=Module)
T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')

__Hook1 = Callable[[TModule, T1], T2]
__Hook2 = Callable[[TModule, T1, T2], T3]

class __RemovableHandle:
    # torch.utils.hooks.RemovableHandle
    def remove(self) -> None: ...


class Parameter(Tensor):
    def __new__(cls,
                data: Optional[Tensor] = None,
                requires_grad: bool = True,
                ) -> Parameter: ...


class Module:
    training: bool

    def __init__(self) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def to(self: TModule, device: Union[device, int, str]) -> TModule: ...
    def apply(self: TModule, fn: Callable[[Module], None]) -> TModule: ...

    def register_buffer(self, name: str, tensor: Tensor) -> None: ...
    def register_parameter(self, name: str, param: Union[Parameter, None]) -> None: ...

    def register_backward_hook(self, hook: __Hook2) -> __RemovableHandle: ...
    def register_forward_pre_hook(self, hook: __Hook1) -> __RemovableHandle: ...
    def register_forward_hook(self, hook: __Hook2) -> __RemovableHandle: ...

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]: ...
    def buffers(self, recurse: bool = True) -> Iterator[Tensor]: ...
    def modules(self) -> Iterator[Module]: ...
    def children(self) -> Iterator[Module]: ...

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]: ...
    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]: ...
    def named_modules(self, memo: None = None, prefix: str = '') -> Iterator[Tuple[str, Module]]: ...
    def named_children(self) -> Iterator[Tuple[str, Module]]: ...

    def add_module(self, name: str, module: Module) -> None: ...

    def state_dict(self, destination: Optional[str] = ..., prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Tensor]: ...
    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = ...) -> Tuple[List[str], List[str]]: ...

    def train(self: TModule, mode: bool = ...) -> TModule: ...
    def eval(self: TModule) -> TModule: ...


class Sequential(Module):
    @overload
    def __init__(self, args: Dict[str, Module]) -> None: ...

    @overload
    def __init__(self, *args: Module) -> None: ...

    def __iter__(self) -> Iterator[Module]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Module: ...

    def forward(self, input: Any) -> Any: ...


class ModuleList(Module):
    def __init__(self, modules: Iterable[Module]) -> None: ...

    def __iter__(self) -> Iterator[Module]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Module: ...


class Linear(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 ) -> None: ...


class Conv2d(Module):
    in_channels: int
    out_channels: int
    kernel_size: Union[int, Tuple[int, ...]]

    weight: Tensor
    bias: Tensor

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 ) -> None: ...


class BatchNorm2d(Module):
    weight: Tensor
    bias: Tensor

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: Optional[float] = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 ) -> None: ...


class MaxPool2d(Module):
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Optional[int] = None,
                 padding: int = 0,
                 dilation: int = 1,
                 return_indices: bool = False,
                 ceil_mode: bool = False) -> None: ...


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size: Union[int, Tuple[int, ...]]) -> None: ...


class ReLU(Module):
    def __init__(self, inplace: bool = False) -> None: ...

#END
