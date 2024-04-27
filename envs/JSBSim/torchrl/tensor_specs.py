from typing import Optional, Union, Sequence

from dataclasses import dataclass
import numpy as np
import torch
from torchrl.data.tensor_specs import MultiOneHotDiscreteTensorSpec

from torch.distributions import Categorical
@dataclass(repr=False)
class ConvertibleMultiOneHotDiscreteTensorSpec(MultiOneHotDiscreteTensorSpec):
    def __init__(
        self,
        nvec: Sequence[int],
        shape: Optional[torch.Size] = None,
        device=None,
        dtype=torch.bool,
        use_register=False,
        mask: torch.Tensor | None = None,
    ):
         super().__init__(nvec, shape, device, dtype, use_register, mask)
        
    def to_numpy(self, val: torch.Tensor, safe: bool = False) -> np.ndarray:
        if safe:
            if not isinstance(val, torch.Tensor):
                raise NotImplementedError
            self.assert_is_in(val)
        start = stop = 0
        output = np.zeros((*val.shape[:-1], len(self.nvec)))
        for i, space in enumerate(self.space):
            stop += space.n
            output[..., i] = val[..., start:stop].long().argmax(-1).cpu().numpy()
            start = stop
        return output