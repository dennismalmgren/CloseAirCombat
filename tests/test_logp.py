from torchrl.modules.distributions import TanhNormal
from torchrl.modules import MLP
from tensordict.nn.distributions import NormalParamExtractor
import torch
import math

with torch.no_grad():
    net = MLP(in_features = 14, out_features = 10, num_cells=[256, 256], activation_class=torch.nn.ReLU)

    print("Net definition")
    print(net)
    in_data_action = torch.tensor([-0.4985,  0.8140,  0.6497])
    in_data_obs = torch.tensor([1.2451e+00,  1.4851e-03,  3.4298e-03, -2.2442e-03, -4.2063e-03,
         -1.5665e-03,  9.2927e-04,  3.6508e-03,  4.4133e-03,  4.1063e-03,
         -3.0357e-04])
    in_data = torch.cat((in_data_action, in_data_obs))

    #in_data = torch.rand(10) * 2 - 1
    print("in-data: ", in_data)
    out = net(in_data)
    print("out: ", out)
    probs = out.softmax(-1)
    print("out-probs: ", probs)
    K = 2
    nbins = 10

    bias = torch.tensor([-0.1] * K + [-100] * (nbins - K))
    #bias = bias / torch.sum(bias)
    print("defined bias: ", bias)
    print("net bias: ", net[-1].bias)
    net[-1].bias.data = bias
    out2 = net(in_data)
    print("out2: ", out2)
    probs2 = out2.softmax(-1)
    print("out2-probs: ", probs2)
    lsm = out2.log_softmax(-1)
    print("out2-logsoftmax: ", lsm)
    
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