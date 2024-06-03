import torch
from torchrl.modules.distributions.continuous import SafeTanhTransform
from torchrl.modules import TanhNormal
from torch.nn import Linear
from torchrl.modules import MLP
import copy

def run_log_prob(linear1):
    input = torch.zeros(1, 1)
    sampled_action = torch.tensor(([1.0], ))
    # Linear layer
   
    mu_output = linear1(input)
    state_independent_scale = torch.nn.Parameter(torch.tensor(([-3.0],)))
    scale_output = torch.exp(state_independent_scale)

    # transform_dist = SafeTanhTransform()
    # transformed_mean = transform_dist(mu_output)
    # print("Transformed action mean: ", transformed_mean)
    # print("Transformed variance: ", scale_output)
    # print("Sampled action: ", sampled_action)
    #first, try with log prob gradient
    total_dist = TanhNormal(mu_output, scale_output)
    lp = -total_dist.log_prob(sampled_action)
    lp.backward()
    print(list(linear1.parameters())[-1].grad)

def run_mse(linear2):
    input = torch.zeros(1, 1)
    sampled_action = torch.tensor(([1.0], ))
    mu_output = linear2(input)
    state_independent_scale = torch.nn.Parameter(torch.tensor(([-3.0],)))
    scale_output = torch.exp(state_independent_scale)

    #second, try mse loss gradient
    total_dist = TanhNormal(mu_output, scale_output)
    action_inverted = total_dist._t._inverse(sampled_action)
    mse_loss = 0.5 * torch.nn.functional.mse_loss(action_inverted, total_dist.loc, reduction='none')
    mse_loss_scaled = mse_loss / (total_dist.scale **2)
    mse_loss_scaled.backward()
    print(list(linear2.parameters())[-1].grad)

linear1 = MLP(in_features=1, 
            out_features=1, 
            num_cells=[64], 
            activation_class=torch.nn.Tanh)
linear2 = copy.deepcopy(linear1)
run_log_prob(linear1)
run_mse(linear2)
#calculated gradients

