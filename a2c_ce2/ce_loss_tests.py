
import torch

def _construct_support(mean_outputs):
    a_min = torch.tensor(-10.0)
    a_max = torch.tensor(10.0)
    nbins = 101
    bin_dist = (a_max - a_min) / (nbins - 1)
    support = torch.arange(a_min, a_max + bin_dist, step=bin_dist)
    support_index = (mean_outputs.detach() - a_min) // bin_dist
    dynamic_support = support + mean_outputs - support[support_index.long()]
    return dynamic_support

def main():
    print('hi')
    means = torch.tensor([[2.0], [3.3]])
    supports = _construct_support(means)

    a_min = torch.tensor(-10.0)
    a_max = torch.tensor(10.0)
    nbins = 101
    mean_output = torch.tensor(5.77)
    bin_dist = (a_max - a_min) / (nbins - 1)
    print(bin_dist)
    support = torch.arange(a_min, a_max + bin_dist, step=bin_dist)
    support_index = (mean_output.detach() - a_min) // bin_dist
    dynamic_support = support + mean_output - support[support_index.long()]
    print(dynamic_support)
    bins_below = torch.ceil((mean_output - a_min) / bin_dist)
    bins_above = torch.ceil((a_max - mean_output) / bin_dist)
    print(bins_below)
    print(bins_above)
    print(bins_below + bins_above + 1)
    print(bins_above * bin_dist + mean_output)
    print(mean_output - bins_below * bin_dist)
    #results in 102 outputs
    support_above = torch.arange(bins_above + 1) * bin_dist + mean_output
    support_below = -torch.arange(bins_below, 0, -1) * bin_dist + mean_output
    support = torch.cat([support_below, support_above])
    print(mean_output)
    print(support)
    print(len(support))
if __name__=="__main__":
    main()
