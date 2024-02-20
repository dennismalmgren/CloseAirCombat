import sys
import os

import torch
from torch import distributions as D
from torch import nn
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from envs.grid.air_combat_geometry import LatLonNEUConverter

class GridBirthExpectedTargetsModel:
    def __init__(self):
        pass
    
    def predict(self,
                t: int,
                grid: torch.Tensor):
        pass

class UniformGridBirthExpectedTargetsModel(GridBirthExpectedTargetsModel):
    def __init__(self,
                 expected_targets_per_area_and_timestep: torch.Tensor,
                 batch_size = torch.Size([])):
        #intensity as a float only supported for empty batch size
        self.batch_size = batch_size
        self.expected_targets_per_area_and_timestep = expected_targets_per_area_and_timestep.expand(*self.batch_size, 1, 1)

    def predict(self,
                t: int,
                grid: torch.Tensor,
              ):
        if t == 0:
            grid.fill_(0)
            grid += self.expected_targets_per_area_and_timestep

class RightSideGridBirthxpectedTargetsModel(GridBirthExpectedTargetsModel):
    def __init__(self,
                 expected_targets_per_area_and_timestep: torch.Tensor,
                 batch_size = torch.Size([])):
        #intensity as a float only supported for empty batch size
        self.batch_size = batch_size
        self.expected_targets_per_area_and_timestep = expected_targets_per_area_and_timestep.reshape(*self.batch_size, 1, 1)

    def predict(self,
                t: int,
                grid: torch.Tensor,
                ):
        if t == 0:
            grid.fill_(0)
            x, y = grid.shape[-2], grid.shape[-1]
            grid[..., x - 1:, :] = self.expected_targets_per_area_and_timestep

class MotionPredictionModel:
    def __init__(self):
        pass
    
    def predict(self,
                t: int,
                grid_current_expected_targets: torch.Tensor,
                grid_birth_expected_targets: torch.Tensor,
                ):
        pass

class ZeroMotionPredictorModel(MotionPredictionModel):
    def __init__(self,
                    grid_cell_width_meters: torch.Tensor,
                    grid_cell_height_meters: torch.Tensor,
                    width: torch.Tensor,
                    height: torch.Tensor,
                    time_step_size: torch.Tensor, 
                    ps: torch.Tensor, #probability of survival
                    batch_size = torch.Size([]),
                    device = torch.device('cpu')):
        self.batch_size = batch_size
        self.device = device
        self.ps = ps
        self.grid_cell_width_meters = grid_cell_width_meters
        self.grid_cell_height_meters = grid_cell_height_meters
        self.time_step_size = time_step_size
        self.width = width
        self.height = height
    
    def predict(self,
                t: int,
                grid_current_expected_targets: torch.Tensor,
                grid_birth_expected_targets: torch.Tensor,
                ):
        grid_current_expected_targets = grid_birth_expected_targets + self.ps.unsqueeze(-1).unsqueeze(-1) * grid_current_expected_targets
        return grid_current_expected_targets
    
class ConstantMotionPredictorModel(MotionPredictionModel):
    def __init__(self,
                 grid_cell_width_meters: torch.Tensor,
                 grid_cell_height_meters: torch.Tensor,
                 width: torch.Tensor,
                 height: torch.Tensor,
                 assumed_object_velocity: torch.Tensor,
                 time_step_size: torch.Tensor, 
                 p_s: torch.Tensor, #probability of survival
                 batch_size = torch.Size([]),
                 device = torch.device('cpu')):

        self.batch_size = batch_size
        self.device = device
        self.p_s = p_s        
        self.grid_cell_width_meters = grid_cell_width_meters
        self.grid_cell_height_meters = grid_cell_height_meters
        self.assumed_object_velocity = assumed_object_velocity
        self.time_step_size = time_step_size
        self.width = width
        self.height = height
        #Velocity vector
        assumed_object_speed = torch.norm(assumed_object_velocity, dim = -1)
        #Acceleration variance
        sigma_w_2 = (0.05*assumed_object_speed)**2
        F_theta = torch.eye(2, device=device)
        F_theta_phi = self.time_step_size * torch.eye(2, dtype=torch.float32, device=device)
        accel_mat = torch.tensor([[0.5*self.time_step_size**2],[self.time_step_size]], device=device)
        Q_theta = sigma_w_2 * (accel_mat @ accel_mat.t())

        #The uncertainty in velocity is of equal magnitude as the speed,
        #in both dimensions.
        P = torch.eye(2, device=device) * assumed_object_speed
        mean_dist = F_theta_phi @ self.assumed_object_velocity
        dist_var = F_theta_phi @ P @ F_theta_phi.t() + Q_theta
        distrib = D.MultivariateNormal(mean_dist, dist_var, validate_args=False)
        std = torch.sqrt(distrib.variance)
        cutoff_stdevs = 3
        kernel_size_x = int(math.ceil(cutoff_stdevs * std[0] / self.grid_cell_width_meters)) * 2 + 1
        kernel_size_y = int(math.ceil(cutoff_stdevs * std[1] / self.grid_cell_height_meters)) * 2 + 1
    
        x = torch.arange(-kernel_size_x // 2 + 1, kernel_size_x // 2 + 1, dtype=torch.float32, device=device) * self.grid_cell_width_meters
        y = torch.arange(-kernel_size_y // 2 + 1, kernel_size_y // 2 + 1, dtype=torch.float32, device=device) * self.grid_cell_height_meters
        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
        xy_grid = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)
        xy_grid = xy_grid.unsqueeze(-1)
        kernel_center_loc = torch.zeros((2, 1), device=device)
        dists = kernel_center_loc - F_theta @ xy_grid
        dists = dists.squeeze(-1)

        log_prob_kernel_values = distrib.log_prob(dists).view(kernel_size_x, kernel_size_y)
        prob_kernel_values = torch.exp(log_prob_kernel_values) 
        prob_kernel_values = prob_kernel_values / prob_kernel_values.sum() #normalize them, for now. 
        prob_kernel_values = prob_kernel_values * self.p_s
        conv_kernel = prob_kernel_values[None, None, :, :]
        self.conv2d_layer = nn.Conv2d(in_channels=1, out_channels=1, 
                         kernel_size=conv_kernel.shape[-1], 
                         bias=False,
                         padding=(conv_kernel.shape[-2] // 2, conv_kernel.shape[-1] // 2), 
                         device=device)
        self.conv2d_layer.weight.data = conv_kernel
        self.conv2d_layer.weight.requires_grad = False

    def predict(self,
                t: int,
                grid_current_expected_targets: torch.Tensor,
                grid_birth_expected_targets: torch.Tensor):
        grid_current_expected_targets = grid_birth_expected_targets + self.conv2d_layer(grid_current_expected_targets)
        return grid_current_expected_targets
    
class GridPPP:
    """
    This class defines an H x W grid, approximately corresponding to 
    The structure corresponds to the paper
    https://liu.diva-portal.org/smash/get/diva2:1637518/FULLTEXT01.pdf
    pd is the probability of detection which is assumed to be constant
    across time.
    ps is the probability of survival which is assumed to be constant
    across time. This allows the intensity of undetected targets to
    saturate

    Grid tasks deal with up to three simultaneous coordinate systems, depending on the 
    simulation environment. To avoid conflicts we use clear naming conventions.
    
    Grid coordinate system (GRID)
    ======================
    Grid is organized as a 2D discretized array, of dimensions (H, W). The lower left corner is (0, 0) and the top right corner is (H-1, W-1).
    The coordinates are indicated by lower case letters (h, w).

    Local euclidean coordinate system (NEU)
    ================================
    This is a 3D coordinate system in meters, using the NEU (North, East, Up) convention, of dimension (Y, X, Z). 
    The origin of the local euclidean coordinate system corresponds to the bottom left corner of the (0, 0) grid cell in the Grid Coordinate System.
    The coordinates are indicated by lower case letters (y, x, z).

    Global elliptic coordinate system (LLA)
    ================================
    This is a 3D coordinate system in degrees and meters, using the LLA (Latitude, Longitude, Altitude) convention. 
    The coordinates are indicated by lower case letters (lat, lon, alt).For a specific scenario, 
    a designated anchor point is chosen, with a specific latitude, longitude and altitude (lat0, lon0, alt0).    
    The anchor point of the global elliptic coordinate system corresponds to the bottom left corner of the (0, 0) grid cell 
    in the Grid Coordinate System, or correspondingly, to the origin of the local euclidean coordinate system.

    Naming conventions for distances and areas
    ==========================================
    To describe distances and areas in the grid coordinate system, we use the suffix _grid. This includes width_grid, height_grid and so on.
    To describe distances and areas in the euclidean coordinate system, we use no suffix. Meters are assumed.
    We avoid describing distances and areas in the global elliptic coordinate system. We only use coordinates. 
    For measures, we transform to a local euclidean system in which we measure, and then back.
    """
    def __init__(self,
                 H: torch.Tensor,
                 W: torch.Tensor,
                 height_meters: float,
                 width_meters: float,
                 ps: torch.Tensor,
                 pd: torch.Tensor,
                 grid_birth_expected_targets_model: GridBirthExpectedTargetsModel,
                 grid_motion_predictor_model: MotionPredictionModel,
                 *,
                 device = torch.device('cpu'),
                 batch_size: torch.Size = torch.Size([])
                 ):
        self.batch_size = batch_size
        self.device = device
        self.H = H
        self.W = W
        self.height_meters = torch.tensor(height_meters, device=device)
        self.width_meters = torch.tensor(width_meters, device=device)
        self.cell_height = self.height_meters / self.H
        self.cell_width = self.width_meters / self.W
        self.cell_w = torch.arange(0, self.W, dtype = torch.float32, device = self.device) + 0.5
        self.cell_h = torch.arange(0, self.H, dtype = torch.float32, device = self.device) + 0.5
        self.cell_w *= self.cell_width
        self.cell_h *= self.cell_height
        self.hh, self.ww = torch.meshgrid(torch.arange(0, self.H, dtype = torch.int, device = self.device), 
                                          torch.arange(0, self.W, dtype = torch.int, device = self.device), indexing="ij")
        self.hh = self.hh.expand((*self.batch_size, H, W))
        self.ww = self.ww.expand((*self.batch_size, H, W))
        #get the mask.
        self.zero_val = torch.tensor(0, device=self.device)
        self.one_val = torch.tensor(1, device=self.device)
        #just need one.
        if len(self.batch_size) > 0:
            self._grid_birth_expected_targets = torch.zeros((1, H, W), dtype = torch.float32, device=device)
        else:
            self._grid_birth_expected_targets = torch.zeros((H, W), dtype = torch.float32, device=device)
        self._grid_current_expected_targets = torch.zeros((*batch_size, H, W), dtype = torch.float32, device=device)

        #if ps.shape != batch_size:
        #    ps = ps.expand(*batch_size)
        #if pd.shape != batch_size:
        #    raise Exception("pd should have the same shape as batch_size")
        self.ps = ps
        #sensor model should also be a convolution that we soar across the grid.
        self.pd = pd
        self.grid_birth_expected_targets_model = grid_birth_expected_targets_model
        self.grid_motion_predictor_model = grid_motion_predictor_model

    @property
    def birth_expected_targets(self):
        return self._grid_birth_expected_targets
    
    @property 
    def current_expected_targets(self):
        return self._grid_current_expected_targets
    
    def predict(self, t: int):
        self.grid_birth_expected_targets_model.predict(t, self.birth_expected_targets)
        self.grid_motion_predictor_model.predict(t, self.current_expected_targets, self.birth_expected_targets)

    def update(self, t: int, sensor_mask: torch.Tensor):
        #it's probable that pd should be an argument here.
        #sensor_mask should be a tensor of shape (batch_size, H, W) with dtype torch.bool
        self._grid_current_expected_targets[sensor_mask] *= (1 - self.pd.squeeze(-1)) 

    def reset(self):
        self.grid_birth_expected_targets_model.predict(0, self.birth_expected_targets)

    def get_square_centered_sensor_coverage_mask_grid(self, location: torch.Tensor, sensor_range: torch.Tensor):
        #given in grid coordinates
        #assumes that the sensor_range is of batch_size.
        #uses grid coordinates. a sensor range of 0 indicates just the location is covered.
        y = location[..., 0]
        x = location[..., 1]        
        mask = torch.zeros((*self.batch_size, self.H, self.W), dtype = torch.bool, device = self.device)
        y_min = torch.max(self.zero_val, y - sensor_range)
        y_max = y + sensor_range + self.one_val
        x_min = torch.max(self.zero_val, x - sensor_range)
        x_max = x + sensor_range + self.one_val
        y_min = y_min.reshape((*self.batch_size, 1, 1))
        y_max = y_max.reshape((*self.batch_size, 1, 1))
        x_min = x_min.reshape((*self.batch_size, 1, 1))
        x_max = x_max.reshape((*self.batch_size, 1, 1))
        mask = (self.hh >= y_min) & (self.hh < y_max) & (self.ww >= x_min) & (self.ww < x_max)
        return mask
    
    def get_square_centered_sensor_coverage_mask(self, location: torch.Tensor, sensor_range_meters: torch.Tensor):
        #assumes that the sensor_range is of batch_size.
        #convert meters to grid coordinates.
        y = location[..., 0]
        x = location[..., 1]
        y = y / self.cell_height
        x = x / self.cell_width
        sensor_range = sensor_range_meters / self.cell_width
        sensor_range = sensor_range.to(torch.int)
        #get the grid coordinates.
        y = y.to(torch.int)
        x = x.to(torch.int)

        mask = torch.zeros((*self.batch_size, self.H, self.W), dtype = torch.bool, device = self.device)

        y_min = torch.max(self.zero_val, y - sensor_range)
        y_max = y + sensor_range + self.one_val
        x_min = torch.max(self.zero_val, x - sensor_range)
        x_max = x + sensor_range + self.one_val
        y_min = y_min.reshape((*self.batch_size, 1, 1))
        y_max = y_max.reshape((*self.batch_size, 1, 1))
        x_min = x_min.reshape((*self.batch_size, 1, 1))
        x_max = x_max.reshape((*self.batch_size, 1, 1))

        mask = (self.hh >= y_min) & (self.hh < y_max) & (self.ww >= x_min) & (self.ww < x_max)
        return mask




def get_NEU_Z_from_LLA(lat: torch.Tensor, lon: torch.Tensor, altitude: torch.Tensor):
    converter = LatLonNEUConverter(torch.device('cpu'), lat, lon, altitude)
    neu = converter.LLA2NEU(lat.item(), lon.item(), altitude.item())
    return neu[2]

class GridLatLonMapper:
    def __init__(self, 
                 grid_patrol_task: GridPPP,
                 lat0: torch.Tensor,
                 lon0: torch.Tensor,
                 alt0: torch.Tensor,
):
        self.grid_patrol_task = grid_patrol_task
        self.lat0 = lat0
        self.lon0 = lon0
        self.alt0 = alt0

        self.cell_width = self.width / grid_patrol_task.W
        self.cell_height = self.height / grid_patrol_task.H
        self.z0 = get_NEU_Z_from_LLA(self.lat0, self.lon0, self.alt0)
        self.converter = LatLonNEUConverter(torch.device('cpu'), self.lat0, self.lon0, self.alt0)
        self.grid_cell_centers_x = torch.arange(0, grid_patrol_task.W, dtype = torch.float32) + 0.5
        self.grid_cell_centers_x *= self.cell_width
        self.grid_cell_centers_y = torch.arange(0, grid_patrol_task.H, dtype = torch.float32) + 0.5
        self.grid_cell_centers_y *= self.cell_height

    
    def get_grid_from_lla(self, lat: torch.Tensor, lon: torch.Tensor, alt: torch.Tensor):
        """
        First convert to local euclidean. 
        Then map to grid coordinates.
        """
        neu = self.converter.LLA2NEU(lat.item(), lon.item(), alt.item())
        y, x, _ = neu
        h = (y / self.cell_height).to(torch.int)
        w = (x / self.cell_width).to(torch.int)
        return h, w
    
    def get_lla_from_grid(self, h: torch.Tensor, w: torch.Tensor):
        y = (h + 0.5) * self.cell_height
        x = (w + 0.5) * self.cell_width
        #we need to equip this with a particular altitude.
        #we use z0 for this.
        lat, lon, alt = self.converter.NEU2LLA(y.item(), x.item(), self.z0.item())
        return lat, lon, alt
    

        #find all the covered grid cell points.

