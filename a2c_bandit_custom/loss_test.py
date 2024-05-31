import torch
from torchrl.modules.distributions.continuous import SafeTanhTransform
from torchrl.modules import TanhNormal
from torch.nn import Linear
from tensordict.nn.utils import biased_softplus

transform_dist = SafeTanhTransform()


input = torch.zeros(1, 1)
sampled_action = torch.tensor(([1.0], ))
# Linear layer
linear = Linear(1, 1)
mu_output = linear(input)
state_independent_scale = torch.nn.Parameter(torch.tensor(([0.0],)))
scale_output = biased_softplus(1.0)(state_independent_scale)

transformed_mean = transform_dist(mu_output)
print("Transformed action mean: ", transformed_mean)
print("Transformed variance: ", scale_output)
print("Sampled action: ", sampled_action)
#first, try with log prob gradient
total_dist = TanhNormal(mu_output, scale_output)
lp = total_dist.log_prob(sampled_action)
optim = torch.optim.Adam(linear.parameters())
optim.zero_grad()
lp.backward()
print(list(linear.parameters())[-1].grad)
optim.zero_grad()

mu_output = linear(input)
tstate_independent_scale = torch.nn.Parameter(torch.zeros_like(mu_output))
scale_output = biased_softplus(1.0)(state_independent_scale)

#second, try mse loss gradient
total_dist = TanhNormal(mu_output, scale_output)
action_inverted = total_dist._t._inverse(sampled_action)
mse_loss = -torch.nn.functional.mse_loss(action_inverted, total_dist.loc, reduction='none')
mse_loss_scaled = mse_loss / (2 * total_dist.scale **2)
mse_loss_scaled.backward()
print(list(linear.parameters())[-1].grad)

#calculated gradients
calc_grad = (action_inverted - total_dist.loc) /(total_dist.scale **2 * (1 - sampled_action**2))
print(calc_grad)
