from torchrl.modules.distributions import TanhNormal

from tensordict.nn.distributions import NormalParamExtractor
import torch
import math
print(math.log(1/2))
actor_extractor = NormalParamExtractor(
    scale_mapping=f"biased_softplus_1.0",
    scale_lb=0.1,
)

input = torch.tensor([[1.0, -3], [1.0, -3], [1.0, -3]])

output = actor_extractor(input)

print(output)
#then we do the tanh_normal
#maximum logp seems to be -3386.7615 for 1-D
#for 2-D it's -6773.5229
#for 3-D which is what we have, it's -10160.2844
#this is calculated by:
dist_class = TanhNormal(loc = output[0], scale=output[1])
action = torch.tensor([[-1.0, -1.0, -1.0]])
logp = dist_class.log_prob(action)
print(logp)
K = 2
nbins = 10

bias = torch.tensor([-0.1] * K + [-100] * (nbins - K))
print(bias)
bias = bias.softmax(-1)
print(bias)