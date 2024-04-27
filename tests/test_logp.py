from torchrl.modules.distributions import TanhNormal
from torchrl.modules import MLP
from tensordict.nn.distributions import NormalParamExtractor
import torch
import math
from torch.distributions import Normal
import numpy as np

#Let's do four 
action_dim = 1
action_min = torch.tensor([-1.0] * action_dim)
action_max = torch.tensor([1.0] * action_dim)

actor_extractor = NormalParamExtractor(
    scale_mapping=f"biased_softplus_1.0",
    scale_lb=0.1,
)
#So the cases we want to study is 

#### mean of 0, minimal variance
#the first value is the action-dim mean, the second is the action-dim std pre softplus.
input = torch.tensor([[1.0, -3]] * action_dim)

#input = torch.tensor([[0.0, -3], [0.0, -3]])

output = actor_extractor(input)
print(-math.log(10_000_000* math.sqrt(2 * math.pi)))
print(-math.log(0.1* math.sqrt(2 * math.pi)))

print(math.log(10e-9))
print(output)
loc = output[0]
scale = output[1]
action_val = -0.5
dist_class = TanhNormal(loc = output[0], scale=output[1])
dist_class_norm = Normal(loc = output[0], scale=output[1]) 
u_val = np.arctanh(action_val)
logp_verify = - math.log(scale * math.sqrt(2 * math.pi)) - 0.5 * ((u_val - loc) / scale)**2 - math.log(1 - math.tanh(u_val)**2)
action = torch.tensor([action_val])
logp = dist_class.log_prob(action)
print("logp: ", logp)   
print("logp_verify: ", logp_verify)
some_act = dist_class.sample()
some_act_logp = dist_class.log_prob(some_act)

action = torch.tensor([[-1.0, -1.0]])
logp = dist_class.log_prob(action)
print("Worst case logp: ", logp)
prob_of_action = math.exp(logp)

action = torch.tensor([[0.0, 0.0]])
logp = dist_class.log_prob(action)
print("Best case logp: ", logp)
# K = 2
# nbins = 10
Q_min = 0
Q_max = 100

# bias = torch.tensor([-0.1] * K + [-100] * (nbins - K))
# print(bias)
# bias = bias.softmax(-1)
# print(bias)