
# from torchrl.modules.distributions import TanhNormal
# from torchrl.modules import MLP
# from tensordict.nn.distributions import NormalParamExtractor
# import torch
# import math
# from tensordict.nn import TensorDictModule, TensorDictSequential
# from tensordict import TensorDict
# import matplotlib.pyplot as plt

# qvalue_net_kwargs = {
#         "num_cells": [512, 512],
#         "out_features": 90,
#         "activation_class": torch.nn.ReLU,
#     }

# qvalue_net = MLP(
#     **qvalue_net_kwargs,
# )

# qvalue1 = TensorDictModule(
#     in_keys=["observation"],
#     out_keys=["state_action_value"],
#     module=qvalue_net,
# )

# qvalue2 = TensorDictModule(lambda x: x.log_softmax(-1), ["state_action_value"], ["state_action_value"])

# #qvalue = TensorDictSequential(qvalue1, qvalue2)
# qvalue = TensorDictSequential(qvalue1)

# optim = torch.optim.Adam(qvalue.parameters(), lr=1e-3)

# in_data = torch.tensor([0.0])
# label_generator_1 = torch.distributions.Normal(-1.0, 0.5)
# label_generator_2 = torch.distributions.Normal(1.0, 0.5)
# in_data = torch.tensor([0.0]).expand((256, 1))
# in_data_td = TensorDict({"observation": in_data}, batch_size=256)
# Vmin = -4.0
# Vmax = 4.0
# nbins = 90
# support = torch.linspace(Vmin, Vmax, nbins)
# atoms = support.numel()
# Vmin = support.min()
# Vmax = support.max()
# delta_z = (Vmax - Vmin) / (atoms - 1)
# stddev = 0.75 * delta_z
# support_plus = support + delta_z / 2
# support_minus = support - delta_z / 2


# for i in range(10000):
#     label_1 = label_generator_1.sample((128,))
#     label_2 = label_generator_2.sample((128,))
#     label = torch.cat([label_1, label_2], dim=0)
#     label = label[torch.randperm(label.size(0))]
#     label = label.unsqueeze(-1)
#     dist = torch.distributions.Normal(label, stddev)
#     cdf_plus = dist.cdf(support_plus)
#     cdf_minus = dist.cdf(support_minus)
#     m = cdf_plus - cdf_minus
#     #m[..., 0] = cdf_plus[..., 0]
#     #m[..., -1] = 1 - cdf_minus[..., -1]
#     m = m / m.sum(dim=-1, keepdim=True) 

#     #randomize order
#     optim.zero_grad()
#     out = qvalue(in_data_td)
# #    loss = torch.nn.functional.mse_loss(out["state_action_value"].mean(-1), label)
#     loss = torch.nn.functional.cross_entropy(out["state_action_value"], m)
#     loss.backward()
#     optim.step()
#     if i % 100 == 0:
#         print("Loss: ", loss.item())
#         mean_pred = out["state_action_value"].softmax(-1) @ support
#         print("Mean Pred: ", mean_pred.mean())
#         print("Out: ", out["state_action_value"].softmax(-1).mean(0))
#         #print("Sorted: ", out["state_action_value"].softmax(-1).mean(0).argsort())
#         #print("Label: ", label)
#         print("")

# print("Final results")
# print("Mean Pred: ", mean_pred.mean())
# print("Out: ", out["state_action_value"].softmax(-1).mean(0))
# print("Support: ", support)
# plt.plot(support, out["state_action_value"].softmax(-1).mean(0).detach().numpy())
# plt.show()