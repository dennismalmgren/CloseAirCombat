from __future__ import annotations

from numbers import Number
from typing import Sequence, Union

import numpy as np

import torch
from tensordict.nn.utils import mappings
from torch import distributions as D, nn

class AddStateIndependentNormalScale(torch.nn.Module):
    """A nn.Module that adds trainable state-independent scale parameters.

    The scale parameters are mapped onto positive values using the specified ``scale_mapping``.

    Args:
        scale_mapping (str, optional): positive mapping function to be used with the std.
            default = "biased_softplus_1.0" (i.e. softplus map with bias such that fn(0.0) = 1.0)
            choices: "softplus", "exp", "relu", "biased_softplus_1";
        scale_lb (Number, optional): The minimum value that the variance can take. Default is 1e-4.

    Examples:
        >>> from torch import nn
        >>> import torch
        >>> num_outputs = 4
        >>> module = nn.Linear(3, num_outputs)
        >>> module_normal = AddStateIndependentNormalScale(num_outputs)
        >>> tensor = torch.randn(3)
        >>> loc, scale = module_normal(module(tensor))
        >>> print(loc.shape, scale.shape)
        torch.Size([4]) torch.Size([4])
        >>> assert (scale > 0).all()
        >>> # with modules that return more than one tensor
        >>> module = nn.LSTM(3, num_outputs)
        >>> module_normal = AddStateIndependentNormalScale(num_outputs)
        >>> tensor = torch.randn(4, 2, 3)
        >>> loc, scale, others = module_normal(*module(tensor))
        >>> print(loc.shape, scale.shape)
        torch.Size([4, 2, 4]) torch.Size([4, 2, 4])
        >>> assert (scale > 0).all()
    """

    def __init__(
        self,
        scale_shape: Union[torch.Size, int, tuple],
        scale_mapping: str = "exp",
        scale_lb: Number = 1e-4,
    ) -> None:

        super().__init__()
        self.scale_lb = scale_lb
        if isinstance(scale_shape, int):
            scale_shape = (scale_shape,)
        self.scale_shape = scale_shape
        self.scale_mapping = scale_mapping
        self.state_independent_scale = torch.nn.Parameter(torch.ones(scale_shape) * torch.log(torch.tensor(0.1)))

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        loc, *others = tensors

        if self.scale_shape != loc.shape[-len(self.scale_shape) :]:
            raise RuntimeError(
                f"Last dimensions of loc ({loc.shape[-len(self.scale_shape):]}) do not match the number of dimensions "
                f"in scale ({self.state_independent_scale.shape})"
            )

        scale = torch.zeros_like(loc) + self.state_independent_scale
        scale = mappings(self.scale_mapping)(scale).clamp_min(self.scale_lb)

        return (loc, scale, *others)
