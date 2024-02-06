import sys
import os

import torch
from torch.distributions import Uniform, Normal

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from envs.grid.grid_ppp import ConstantExpectedTargetsModel, ConstantSurvivalProbabilityModel, GridPPP

class TestPPP:
    def test_env(self):
        #2-D grid N x M
        #grid cells are square
        #starting at (0, 0)
        H = 2
        W = 3
        N = H * W

        grid_cell_size = 100
        grid_points_h = grid_cell_size / 2 + grid_cell_size * torch.arange(0, H)
        grid_points_w = grid_cell_size / 2 + grid_cell_size * torch.arange(0, W)
        hh, ww = torch.meshgrid(grid_points_h, grid_points_w, indexing='ij')

        grid_coordinates = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=0)
        #dim = 2xHW
        h = 1
        w = 1
        index = h * W + w
        print(grid_coordinates[:, index])

        #with uniform intensities, nothing ever happens.
        #we need to juice it up.
        
        grid_intensities = torch.ones(H, W) #lets do it uniform.

        print('ok')
#        expected_targets_model = ConstantExpectedTargetsModel(torch.ones(H, W))
#        survival_probability_model = ConstantSurvivalProbabilityModel(torch.ones(H, W))

#        theta = Uniform(low = torch.zeros((H, W)), high = torch.ones((N, M))
#        phi_prior = torch.ones(H, W)
#        env = GridPPP(expected_targets_model, survival_probability_model, theta, phi_prior)
