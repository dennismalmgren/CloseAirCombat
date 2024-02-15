import sys
import os

import torch
from torch.distributions import Uniform, Normal
import pytest


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from envs.grid.grid_ppp import GridPPP, UniformNoMovementGridBirthIntensityInitialization
#40 mil är 400 km.
#20 mil är 200 km.
# vi vill alltså jobba med 400x200km-rutor senare.
class TestPPP:
    def test_grid(self):
        H = 20
        W = 40
        width = 40_000 #takes about 20 seconds in mach 1.5
        height = 20_000
        ps = torch.tensor(0.99)
        pd = torch.tensor(0.9)
        intensity = torch.tensor(0.1)
        birthintensitymodel = UniformNoMovementGridBirthIntensityInitialization(intensity)

        grid_ppp = GridPPP(H, W, height, width, ps, pd, birthintensitymodel)
        grid_ppp.predict()
        assert grid_ppp.current_intensity_grid.shape == (H, W)
        assert grid_ppp.current_intensity_grid[0, 0] == intensity
        sensor_mask = torch.ones(H, W, dtype = torch.bool)
        grid_ppp.update(sensor_mask)
        assert grid_ppp.current_intensity_grid[0, 0] == pytest.approx(torch.tensor(0.01))#intensity * (1 - pd)

    def test_grid_batched(self):
            H = 20
            W = 40
            width = 40_000 #takes about 20 seconds in mach 1.5
            height = 20_000
            B = 10
            batch_size = torch.Size([B])
            intensity_base = torch.tensor(0.1)
            intensity = intensity_base * torch.ones(batch_size)
            ps = torch.tensor(0.99) * torch.ones(batch_size)
            pd = torch.tensor(0.9) * torch.ones(batch_size)
            birthintensitymodel = UniformNoMovementGridBirthIntensityInitialization(intensity,
                                                                                    batch_size=batch_size)

            grid_ppp = GridPPP(H, W, height, width, ps, pd, birthintensitymodel,
                               batch_size=batch_size)
            
            grid_ppp.predict()
            assert grid_ppp.current_intensity_grid.shape == (B, H, W)
            assert grid_ppp.current_intensity_grid[0, 0, 0] == pytest.approx(intensity_base)
            sensor_mask = torch.ones((B, H, W), dtype = torch.bool)
            grid_ppp.update(sensor_mask)
            assert grid_ppp.current_intensity_grid[0, 0, 0] == pytest.approx(torch.tensor(0.01))#intensity * (1 - pd)

    def test_grid_cuda(self):
        device = torch.device('cuda')
        H = 20
        W = 40
        height = 20_000
        width = 40_000 #takes about 20 seconds in mach 1.5
        ps = torch.tensor(0.99, device=device)
        pd = torch.tensor(0.9, device=device)
        intensity = torch.tensor(0.1, device=device)
        birthintensitymodel = UniformNoMovementGridBirthIntensityInitialization(intensity)

        grid_ppp = GridPPP(H, W, height, width, ps, pd, birthintensitymodel, device = device)
        grid_ppp.predict()
        assert grid_ppp.current_intensity_grid.shape == (H, W)
        assert grid_ppp.current_intensity_grid[0, 0] == intensity
        sensor_mask = torch.ones(H, W, dtype = torch.bool, device = device)
        grid_ppp.update(sensor_mask)
        assert grid_ppp.current_intensity_grid[0, 0].cpu().numpy() == pytest.approx(torch.tensor(0.01, device = device).cpu().numpy())#intensity * (1 - pd)