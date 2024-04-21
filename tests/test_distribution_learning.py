
from torchrl.modules.distributions import TanhNormal
from torchrl.modules import MLP
from tensordict.nn.distributions import NormalParamExtractor
import torch
import math
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict import TensorDict

qvalue_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": torch.nn.ReLU,
    }

qvalue_net = MLP(
    **qvalue_net_kwargs,
)

qvalue1 = TensorDictModule(
    in_keys=["observation"],
    out_keys=["state_action_value"],
    module=qvalue_net,
)

qvalue2 = TensorDictModule(lambda x: x.log_softmax(-1), ["state_action_value"], ["state_action_value"])

#qvalue = TensorDictSequential(qvalue1, qvalue2)
qvalue = TensorDictSequential(qvalue1)

optim = torch.optim.Adam(qvalue.parameters(), lr=1e-3)

in_data = torch.tensor([0.0])
label_generator = torch.distributions.Normal(0.0, 1)
in_data = torch.tensor([0.0]).expand((256, 1))
in_data_td = TensorDict({"observation": in_data}, batch_size=256)
for i in range(10000):
    label = label_generator.sample((256,))
    optim.zero_grad()
    out = qvalue(in_data_td)
    loss = torch.nn.functional.mse_loss(out["state_action_value"].mean(-1), label)
    loss.backward()
    optim.step()
    if i % 100 == 0:
        print("Loss: ", loss.item())
        print("Out: ", out["state_action_value"].mean())
        #print("Label: ", label)
        print("")