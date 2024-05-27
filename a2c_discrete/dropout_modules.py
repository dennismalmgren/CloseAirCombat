
from torch import nn
from tensordict.nn import TensorDictModuleBase
from tensordict import NestedKey
import torch

class ConsistentDropout(nn.Module):
    """Consistent Dropout.

    This module was proposed in "Consistent Dropout for Policy Gradient Reinforcement Learning"
    (Matthew Hausknecht, Nolan Wagener), https://arxiv.org/abs/2202.11818

    This code is a light modification of the original code proposed in the paper, to make
    it well fit for carrying data via TensorDict primitives.

    """

    def __init__(self, p=0.5): #inplace?
        super().__init__()
        self.p = p
        self.scale_factor = 1 / (1 - self.p)

    def forward(self, x, mask=None):
        if self.training:
            if mask is None:
                mask = self.make_mask(x)
            return x * mask * self.scale_factor, mask
        else:
            return x

    def make_mask(self, x):
        return torch.bernoulli(torch.full_like(x, 1 - self.p)).bool()

class ConsistentDropoutModule(TensorDictModuleBase):
    """

    Examples:
        >>> from tensordict import TensorDict
        >>> consistent_dropout = ConsistentDropout()
        >>> module = ConsistentDropoutModule(consistent_dropout)
        >>> td = TensorDict({"x": torch.randn(3, 4)}, [3])
        >>> module(td)
        TensorDict(
            fields={
                mask_6127171760: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.bool, is_shared=False),
                x: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    """
    def __init__(self, consistent_dropout: ConsistentDropout, in_key: NestedKey=None, in_keys=None, out_keys=None):
        if in_key is None:
            in_key = "x"
        if in_keys is None:
            in_keys = [in_key, f"mask_{id(self)}"]
        elif len(in_keys) != 2:
            raise ValueError("in_keys and out_keys length must be 2 for consistent dropout.")
        if out_keys is None:
            out_keys = [in_key, f"mask_{id(self)}"]
        elif len(out_keys) != 2:
            raise ValueError("in_keys and out_keys length must be 2 for consistent dropout.")
        self.in_keys = in_keys
        self.out_keys = out_keys
        super().__init__()
        self.consistent_dropout = consistent_dropout

    def forward(self, tensordict):
        x = tensordict.get(self.in_keys[0])
        mask = tensordict.get(self.in_keys[1], default=None)
        x, mask = self.consistent_dropout(x, mask=mask)
        tensordict.set(self.out_keys[0], x)
        tensordict.set(self.out_keys[1], mask)
        return tensordict