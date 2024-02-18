import sys
import os

import torch
from torch import vmap

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from envs.grid.air_combat_geometry import LatLonNEUConverter

class GridBirthIntensityInitialization:
    def __init__(self):
        pass

class UniformNoMovementGridBirthIntensityInitialization(GridBirthIntensityInitialization):
    def __init__(self,
                 intensity: torch.Tensor,
                 batch_size = torch.Size([])):
        #intensity as a float only supported for empty batch size
        self.batch_size = batch_size
        self.intensity = intensity.reshape(*self.batch_size, 1, 1)

    #TODO: Later there will be a t argument such that the birth intensity can change over the course of the scenario.
    def apply(self,
              grid: torch.Tensor):
        grid.fill_(0)
        grid += self.intensity

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
                 H: int,
                 W: int,
                 height: float,
                 width: float,
                 ps: torch.Tensor,
                 pd: torch.Tensor,
                 birth_intensity_initialization: GridBirthIntensityInitialization,
                 *,
                 device = torch.device('cpu'),
                 batch_size: torch.Size = torch.Size([])
                 ):
        self.batch_size = batch_size
        self.device = device
        self.H = torch.tensor(H, device=device)
        self.W = torch.tensor(W, device=device)
        self.height = torch.tensor(height, device=device)
        self.width = torch.tensor(width, device=device)
        self.cell_height = self.height / self.H
        self.cell_width = self.width / self.W
        self.cell_w = torch.arange(0, self.W, dtype = torch.float32, device = self.device) + 0.5
        self.cell_h = torch.arange(0, self.H, dtype = torch.float32, device = self.device) + 0.5
        self.cell_w *= self.cell_width
        self.cell_h *= self.cell_height
        self.cell_ww, self.cell_hh = torch.meshgrid(self.cell_w, self.cell_h)
        self.hh, self.ww = torch.meshgrid(torch.arange(0, self.H, dtype = torch.int, device = self.device), 
                                          torch.arange(0, self.W, dtype = torch.int, device = self.device))
        self.hh = self.hh.expand((*self.batch_size, H, W))
        self.ww = self.ww.expand((*self.batch_size, H, W))
        #get the mask.
        self.zero_val = torch.tensor(0, device=self.device)
        self.one_val = torch.tensor(1, device=self.device)
        self.birth_intensity_grid = torch.zeros((*batch_size, H, W), dtype = torch.float32, device=device)
        self.current_intensity_grid = torch.zeros((*batch_size, H, W), dtype = torch.float32, device=device)
        if ps.shape != batch_size:
            raise Exception("ps should have the same shape as batch_size")
        if pd.shape != batch_size:
            raise Exception("pd should have the same shape as batch_size")
        self.ps = ps.reshape(*batch_size, 1, 1)
        self.pd = pd.reshape(*batch_size, 1, 1).expand_as(self.current_intensity_grid)
        self.birth_intensity_initialization = birth_intensity_initialization
        
    
    def predict(self):
        self.birth_intensity_initialization.apply(self.birth_intensity_grid)
        self.current_intensity_grid = self.ps * self.current_intensity_grid + self.birth_intensity_grid

    def update(self, sensor_mask: torch.Tensor):
        #it's probable that pd should be an argument here.
        #sensor_mask should be a tensor of shape (batch_size, H, W) with dtype torch.bool
        self.current_intensity_grid[sensor_mask] *= (1 - self.pd[sensor_mask]) 

    def reset(self):
        self.birth_intensity_initialization.apply(self.birth_intensity_grid)

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

