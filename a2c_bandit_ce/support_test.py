import torch

class SupportOperator(torch.nn.Module):
    def __init__(self, support):
        super().__init__()
        self.register_buffer("support", support)

    def forward(self, x):
        return (x.softmax(-1) * self.support).sum(-1, keepdim=True)
    

low = torch.tensor([-1, -1, -1], dtype=torch.float32)
high = torch.tensor([1, 1, 1], dtype=torch.float32)
nbins = 101
num_outputs = len(low)

supports = [torch.linspace(low[i], high[i], nbins) for i in range(num_outputs)]
support = torch.stack(supports, dim=0)
operator = SupportOperator(support)

input = torch.zeros(1, num_outputs, nbins)
input[0, :, 0] = 100000.0
output = operator(input)

print('ok')