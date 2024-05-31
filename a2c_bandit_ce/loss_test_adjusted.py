import torch
from torchrl.modules.distributions.continuous import SafeTanhTransform
from torchrl.modules import TanhNormal
from torch.nn import Linear
from tensordict.nn.utils import biased_softplus

transform_dist = SafeTanhTransform()

# Input and linear layer
input = torch.zeros(1, 1)
linear = Linear(1, 1)
mu_output = linear(input)
state_independent_scale = torch.nn.Parameter(torch.zeros_like(mu_output))
mu_out_transformed = transform_dist(mu_output)
scale_out_transformed = biased_softplus(1.0)(state_independent_scale)

sampled_action = torch.tensor([[0.999]])

# First, try with log prob gradient
total_dist = TanhNormal(mu_out_transformed, scale_out_transformed)
lp = total_dist.log_prob(sampled_action)
optim = torch.optim.Adam(linear.parameters())
optim.zero_grad()
lp.backward()
log_prob_grad = list(linear.parameters())[-1].grad.clone()
print("Log prob gradient:", log_prob_grad)
optim.zero_grad()

# Second, try mse loss gradient
mu_output = linear(input)
state_independent_scale = torch.nn.Parameter(torch.zeros_like(mu_output))
mu_out_transformed = transform_dist(mu_output)
scale_out_transformed = biased_softplus(1.0)(state_independent_scale)

total_dist = TanhNormal(mu_out_transformed, scale_out_transformed)
action_inverted = total_dist._t._inverse(sampled_action)

# Compute the correct scaled MSE loss
epsilon = 1e-6
mse_loss = torch.nn.functional.mse_loss(action_inverted, total_dist.loc, reduction='none')
# Include the Jacobian term (1 - y^2) correctly
jacobian_term = (1 - sampled_action**2 + epsilon)
mse_loss_scaled = mse_loss / (total_dist.scale**2 * jacobian_term)
mse_loss_scaled = mse_loss_scaled.sum()  # Ensure scalar loss for backward
mse_loss_scaled.backward()
mse_loss_grad = list(linear.parameters())[-1].grad.clone()
print("MSE loss gradient:", mse_loss_grad)

# Compare the gradients
print("Difference in gradients:", log_prob_grad - mse_loss_grad)
