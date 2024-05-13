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

    def test_1d_sensor_mask(self):
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
        sensor_mask = torch.zeros(H, W, dtype = torch.bool)
        sensor_mask[2:4, 2] = True
        grid_ppp.update(sensor_mask)
        assert grid_ppp.current_intensity_grid[2, 2] == pytest.approx(torch.tensor(0.01))#intensity * (1 - pd)
        assert grid_ppp.current_intensity_grid[5, 2] == pytest.approx(intensity)#intensity * (1 - pd)

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
        
    def test_1d_sensor_batched(self):
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
        sensor_mask = torch.zeros((B, H, W), dtype = torch.bool)
        sensor_mask[0, 2:4, 0] = True
        sensor_mask[1, 5, 5] = True
        grid_ppp.update(sensor_mask)
        assert grid_ppp.current_intensity_grid[0, 2, 0] == pytest.approx(torch.tensor(0.01))#intensity * (1 - pd)
        assert grid_ppp.current_intensity_grid[1, 5, 5] == pytest.approx(torch.tensor(0.01))#intensity * (1 - pd)
        assert grid_ppp.current_intensity_grid[4, 5, 5] == pytest.approx(intensity_base)#intensity * (1 - pd)

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
        assert grid_ppp.current_intensity_grid[0, 0].cpu().numpy() == pytest.approx(intensity.cpu().numpy())
        sensor_mask = torch.ones(H, W, dtype = torch.bool, device = device)
        grid_ppp.update(sensor_mask)
        assert grid_ppp.current_intensity_grid[0, 0].cpu().numpy() == pytest.approx(torch.tensor(0.01, device = device).cpu().numpy())#intensity * (1 - pd)


    def test_grid_cuda_batched(self):
        device = torch.device('cuda')
        H = 20
        W = 40
        height = 20_000
        width = 40_000 #takes about 20 seconds in mach 1.5
        B = 10
        batch_size = torch.Size([B])
        ps = 0.99 * torch.ones(batch_size, device=device)
        pd = 0.9 * torch.ones(batch_size, device=device)
        intensity_base = torch.tensor(0.1, device = device)
        intensity = intensity_base * torch.ones(batch_size, device=device)
        birthintensitymodel = UniformNoMovementGridBirthIntensityInitialization(intensity, batch_size=batch_size)

        grid_ppp = GridPPP(H, W, height, width, ps, pd, birthintensitymodel, device = device, batch_size=batch_size)
        grid_ppp.predict()
        assert grid_ppp.current_intensity_grid.shape == (B, H, W)
        assert grid_ppp.current_intensity_grid[0, 0, 0].cpu().numpy() == pytest.approx(intensity_base.cpu().numpy())
        sensor_mask = torch.ones((B, H, W), dtype = torch.bool, device = device)
        grid_ppp.update(sensor_mask)
        assert grid_ppp.current_intensity_grid[0, 0, 0].cpu().numpy() == pytest.approx(torch.tensor(0.01, device = device).cpu().numpy())#intensity * (1 - pd)


    def test_square_centered_sensor_coverage_mask(self):
        H = 20
        W = 40
        width = 40_000 #takes about 20 seconds in mach 1.5
        height = 20_000
        ps = torch.tensor(0.99)
        pd = torch.tensor(0.9)
        intensity = torch.tensor(0.1)
        birthintensitymodel = UniformNoMovementGridBirthIntensityInitialization(intensity)

        grid_ppp = GridPPP(H, W, height, width, ps, pd, birthintensitymodel)
        location = torch.tensor([10_000, 10_000])
        sensor_range_meters = torch.tensor(1_000)
        mask = grid_ppp.get_square_centered_sensor_coverage_mask(location, sensor_range_meters)
        assert mask[9:12, 9:12].all()


    def test_square_centered_sensor_coverage_mask_edge(self):
        H = 20
        W = 40
        width = 40_000 #takes about 20 seconds in mach 1.5
        height = 20_000
        ps = torch.tensor(0.99)
        pd = torch.tensor(0.9)
        intensity = torch.tensor(0.1)
        birthintensitymodel = UniformNoMovementGridBirthIntensityInitialization(intensity)

        grid_ppp = GridPPP(H, W, height, width, ps, pd, birthintensitymodel)
        location = torch.tensor([500, 500])
        sensor_range_meters = torch.tensor(1_000)
        mask = grid_ppp.get_square_centered_sensor_coverage_mask(location, sensor_range_meters)
        assert mask[0:2, 0:2].all()
        sensor_range_meters = torch.tensor(999)
        mask = grid_ppp.get_square_centered_sensor_coverage_mask(location, sensor_range_meters)
        assert mask[0:1, 0:1].all()

    def test_square_centered_sensor_coverage_mask_grid(self):
        H = 20
        W = 40
        width = 40_000 #takes about 20 seconds in mach 1.5
        height = 20_000
        ps = torch.tensor(0.99)
        pd = torch.tensor(0.9)
        intensity = torch.tensor(0.1)
        birthintensitymodel = UniformNoMovementGridBirthIntensityInitialization(intensity)

        grid_ppp = GridPPP(H, W, height, width, ps, pd, birthintensitymodel)
        location = torch.tensor([10, 10], dtype=torch.int32)
        sensor_range = torch.tensor(0, dtype=torch.int32) #covers 1x1 grid cell
        mask = grid_ppp.get_square_centered_sensor_coverage_mask_grid(location, sensor_range)
        assert mask[10, 10]
        assert not mask[11, 11]

        sensor_range = torch.tensor(1, dtype=torch.int32)
        mask = grid_ppp.get_square_centered_sensor_coverage_mask_grid(location, sensor_range)
        assert mask[10, 10]
        assert mask[11, 11]

    def test_square_centered_sensor_coverage_mask_edge_grid(self):
        H = 20
        W = 40
        width = 40_000 #takes about 20 seconds in mach 1.5
        height = 20_000
        ps = torch.tensor(0.99)
        pd = torch.tensor(0.9)
        intensity = torch.tensor(0.1)
        birthintensitymodel = UniformNoMovementGridBirthIntensityInitialization(intensity)

        grid_ppp = GridPPP(H, W, height, width, ps, pd, birthintensitymodel)
        location = torch.tensor([0, 0], dtype=torch.int32)
        sensor_range = torch.tensor(0, dtype=torch.int32) #covers 1x1 grid cell
        mask = grid_ppp.get_square_centered_sensor_coverage_mask_grid(location, sensor_range)
        assert mask[0, 0]
        assert not mask[0, 1]

        sensor_range = torch.tensor(1, dtype=torch.int32)
        mask = grid_ppp.get_square_centered_sensor_coverage_mask_grid(location, sensor_range)
        assert mask[0, 0]
        assert mask[1, 1]

    
    def test_square_centered_sensor_coverage_mask_batched(self):
        H = 20
        W = 40
        width = 40_000 #takes about 20 seconds in mach 1.5
        height = 20_000
        B = 10
        device = torch.device('cpu')
        batch_size = torch.Size([B])
        ps = 0.99 * torch.ones(batch_size, device=device)
        pd = 0.9 * torch.ones(batch_size, device=device)
        intensity_base = torch.tensor(0.1, device = device)
        intensity = intensity_base * torch.ones(batch_size, device=device)
        birthintensitymodel = UniformNoMovementGridBirthIntensityInitialization(intensity, batch_size=batch_size)

        grid_ppp = GridPPP(H, W, height, width, ps, pd, birthintensitymodel, device = device, batch_size=batch_size)
        location_x = 10_000 * torch.ones(batch_size, device=device)
        location_y = 10_000 * torch.ones(batch_size, device=device)
        location = torch.stack([location_x, location_y], dim = -1)
        sensor_range_meters = 1_000 * torch.ones(batch_size)
        mask = grid_ppp.get_square_centered_sensor_coverage_mask(location, sensor_range_meters)
        assert mask[0, 9:12, 9:12].all()


    def test_square_centered_sensor_coverage_mask_edge_batched(self):
        H = 20
        W = 40
        width = 40_000 #takes about 20 seconds in mach 1.5
        height = 20_000
        B = 10
        device = torch.device('cpu')
        batch_size = torch.Size([B])
        ps = 0.99 * torch.ones(batch_size, device=device)
        pd = 0.9 * torch.ones(batch_size, device=device)
        intensity_base = torch.tensor(0.1, device = device)
        intensity = intensity_base * torch.ones(batch_size, device=device)
        birthintensitymodel = UniformNoMovementGridBirthIntensityInitialization(intensity, batch_size=batch_size)

        grid_ppp = GridPPP(H, W, height, width, ps, pd, birthintensitymodel, device = device, batch_size=batch_size)
        location_x = 500 * torch.ones(batch_size, device=device)
        location_y = 500 * torch.ones(batch_size, device=device)
        location = torch.stack([location_x, location_y], dim = -1)
        sensor_range_meters = 1_000 * torch.ones(batch_size)
        mask = grid_ppp.get_square_centered_sensor_coverage_mask(location, sensor_range_meters)
        assert mask[0, 0:2, 0:2].all()
        sensor_range_meters = 999 * torch.ones(batch_size)
        mask = grid_ppp.get_square_centered_sensor_coverage_mask(location, sensor_range_meters)
        assert mask[0, 0:1, 0:1].all()


    def test_square_centered_sensor_coverage_mask_grid_batched(self):
        H = 20
        W = 40
        width = 40_000 #takes about 20 seconds in mach 1.5
        height = 20_000
        B = 10
        device = torch.device('cpu')
        batch_size = torch.Size([B])
        ps = 0.99 * torch.ones(batch_size, device=device)
        pd = 0.9 * torch.ones(batch_size, device=device)
        intensity_base = torch.tensor(0.1, device = device)
        intensity = intensity_base * torch.ones(batch_size, device=device)
        birthintensitymodel = UniformNoMovementGridBirthIntensityInitialization(intensity, batch_size=batch_size)

        grid_ppp = GridPPP(H, W, height, width, ps, pd, birthintensitymodel, device = device, batch_size=batch_size)

        location_x = 10 * torch.ones(batch_size, device=device, dtype=torch.int32)
        location_y = 10 * torch.ones(batch_size, device=device, dtype=torch.int32)
        location = torch.stack([location_x, location_y], dim = -1)
        sensor_range = torch.zeros(batch_size, device=device, dtype=torch.int32)
        mask = grid_ppp.get_square_centered_sensor_coverage_mask_grid(location, sensor_range)
        assert mask[0, 10, 10]
        assert not mask[0, 11, 11]
        sensor_range = torch.ones(batch_size, device=device, dtype=torch.int32)
        mask = grid_ppp.get_square_centered_sensor_coverage_mask_grid(location, sensor_range)
        assert mask[0, 10, 10]
        assert mask[0, 11, 11]

    def test_square_centered_sensor_coverage_mask_edge_grid_batched(self):
        H = 20
        W = 40
        width = 40_000 #takes about 20 seconds in mach 1.5
        height = 20_000
        B = 10
        device = torch.device('cpu')
        batch_size = torch.Size([B])
        ps = 0.99 * torch.ones(batch_size, device=device)
        pd = 0.9 * torch.ones(batch_size, device=device)
        intensity_base = torch.tensor(0.1, device = device)
        intensity = intensity_base * torch.ones(batch_size, device=device)
        birthintensitymodel = UniformNoMovementGridBirthIntensityInitialization(intensity, batch_size=batch_size)

        grid_ppp = GridPPP(H, W, height, width, ps, pd, birthintensitymodel, device = device, batch_size=batch_size)

        location_x = 0 * torch.ones(batch_size, device=device, dtype=torch.int32)
        location_y = 0 * torch.ones(batch_size, device=device, dtype=torch.int32)
        location = torch.stack([location_x, location_y], dim = -1)
        sensor_range = torch.zeros(batch_size, device=device, dtype=torch.int32)
        mask = grid_ppp.get_square_centered_sensor_coverage_mask_grid(location, sensor_range)
        assert mask[0, 0, 0]
        assert not mask[0, 0, 1]
        sensor_range = torch.ones(batch_size, device=device, dtype=torch.int32)
        mask = grid_ppp.get_square_centered_sensor_coverage_mask_grid(location, sensor_range)
        assert mask[0, 0, 0]
        assert mask[0, 1, 1]


    def test_square_centered_sensor_coverage_mask_cuda(self):
        H = 20
        W = 40
        width = 40_000 #takes about 20 seconds in mach 1.5
        height = 20_000
        device = torch.device('cuda')
        ps = torch.tensor(0.99, device=device)
        pd = torch.tensor(0.9, device=device)
        intensity = torch.tensor(0.1, device=device)
        birthintensitymodel = UniformNoMovementGridBirthIntensityInitialization(intensity)

        grid_ppp = GridPPP(H, W, height, width, ps, pd, birthintensitymodel, device = device)
        location = torch.tensor([10_000, 10_000], device=device)
        sensor_range_meters = torch.tensor(1_000, device=device)
        mask = grid_ppp.get_square_centered_sensor_coverage_mask(location, sensor_range_meters)
        assert mask[9:12, 9:12].all()

    def test_square_centered_sensor_coverage_mask_batched_cuda(self):
        H = 20
        W = 40
        width = 40_000 #takes about 20 seconds in mach 1.5
        height = 20_000
        B = 10
        device = torch.device('cuda')
        batch_size = torch.Size([B])
        ps = 0.99 * torch.ones(batch_size, device=device)
        pd = 0.9 * torch.ones(batch_size, device=device)
        intensity_base = torch.tensor(0.1, device = device)
        intensity = intensity_base * torch.ones(batch_size, device=device)
        birthintensitymodel = UniformNoMovementGridBirthIntensityInitialization(intensity, batch_size=batch_size)

        grid_ppp = GridPPP(H, W, height, width, ps, pd, birthintensitymodel, device = device, batch_size=batch_size)
        location_x = 10_000 * torch.ones(batch_size, device=device)
        location_y = 10_000 * torch.ones(batch_size, device=device)
        location = torch.stack([location_x, location_y], dim = -1)
        sensor_range_meters = 1_000 * torch.ones(batch_size, device=device)
        mask = grid_ppp.get_square_centered_sensor_coverage_mask(location, sensor_range_meters)
        assert mask[0, 9:12, 9:12].all()