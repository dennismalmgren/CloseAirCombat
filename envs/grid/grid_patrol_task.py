import torch
from air_combat_geometry import LatLonNEUConverter

class GridBirthIntensityInitialization:
    def __init__(self):
        pass

class UniformNoMovementGridBirthIntensityInitialization(GridBirthIntensityInitialization):
    def __init__(self,
                 intensity: float | torch.Tensor,
                 batch_size = torch.Size([])):
        #intensity as a float only supported for empty batch size
        self.intensity = intensity
        self.batch_size = batch_size

    def apply(self,
              grid: torch.Tensor):
        if isinstance(self.intensity, torch.Tensor):
            grid.fill_(self.intensity)
        grid.fill_(self.intensity)

class GridPatrolTask:
    #This class defines an H x W grid, approximately corresponding to 
    #The structure corresponds to the paper
    #https://liu.diva-portal.org/smash/get/diva2:1637518/FULLTEXT01.pdf
    #pd is the probability of detection which is assumed to be constant
    #across time.
    #ps is the probability of survival which is assumed to be constant
    #across time. This allows the intensity of undetected targets to
    #saturate
    def __init__(self,
                 H: int,
                 W: int,
                 ps: torch.Tensor,
                 pd: torch.Tensor,
                 birth_intensity_initialization: GridBirthIntensityInitialization,
                 batch_size: torch.Size = torch.Size([])
                 ):
        self.batch_size = batch_size
        self.H = H
        self.W = W
        self.birth_intensity_grid = torch.zeros((*batch_size, H, W), dtype = torch.float32)
        self.current_intensity_grid = torch.zeros((*batch_size, H, W), dtype = torch.float32)
        self.ps = ps
        self.pd = pd
        self.birth_intensity_initialization = birth_intensity_initialization
        self.birth_intensity_initialization.apply(self.birth_intensity_grid)
    
    def predict(self):
        self.current_intensity_grid = self.ps * self.current_intensity_grid + self.birth_intensity_grid

    def update(self, sensor_mask: torch.Tensor):
        #sensor_mask should be a tensor of shape (batch_size, H, W) with dtype torch.bool
        self.current_intensity_grid[sensor_mask] *= (1 - self.pd) 

    def reset(self):
        self.birth_intensity_initialization.apply(self.birth_intensity_grid)


#move to its own file with pymap3d dependency
def get_altitude_meters_from_LLA_altitude(lat: torch.Tensor, lon: torch.Tensor, altitude: torch.Tensor):
    converter = LatLonNEUConverter(torch.device('cpu'), lat, lon, altitude)
    neu = converter.LLA2NEU(lat.item(), lon.item(), altitude.item())
    return neu[2]

class GridLatLonMapper:
    #Anchor point is expected to be the top left of the grid, 
    # with latitude across the 'H' axis (height) and longitude across the 'W' axis (width)
    #Since grid cells are represented by their center point, the actual anchor point will not be on the (0, 0) grid cell.
    #width, height and anchor altitude are given in meters
    #tensors are assumed to be on the CPU for now.
    def __init__(self, 
                 grid_patrol_task: GridPatrolTask,
                 anchor_lat: torch.Tensor,
                 anchor_lon: torch.Tensor,
                 anchor_alt: torch.Tensor,
                 width_meters: torch.Tensor,
                 height_meters: torch.Tensor):
        self.grid_patrol_task = grid_patrol_task
        self.anchor_lat = anchor_lat
        self.anchor_lon = anchor_lon
        self.anchor_alt = anchor_alt
        self.width_meters = width_meters
        self.height_meters = height_meters

        self.cell_width = width_meters / grid_patrol_task.W
        self.cell_height = height_meters / grid_patrol_task.H

        self.converter = LatLonNEUConverter(torch.device('cpu'), self.anchor_lat, self.anchor_lon, self.anchor_alt)
        self.altitude_meters = self.converter.NEU2LLA(anchor_lat, anchor_lon, anchor_alt)[2]
        self.grid_centers_w = torch.arange(0, grid_patrol_task.W, dtype = torch.float32) + 0.5
        self.grid_centers_w *= self.cell_width
        self.grid_centers_h = torch.arange(0, grid_patrol_task.H, dtype = torch.float32) + 0.5
        self.grid_centers_h *= self.cell_height

    
    def get_grid_cell_index(self, lat: torch.Tensor, lon: torch.Tensor):
        neu = self.converter.LLA2NEU(lat.item(), lon.item(), self.anchor_alt.item())
        n, e, _ = neu
        h = (n / self.cell_height).to(torch.int)
        w = (e / self.cell_width).to(torch.int)
        return h, w
    
    def get_lat_lon_from_grid_cell(self, h: torch.Tensor, w: torch.Tensor):
        n = (h + 0.5) * self.cell_height
        e = (w  + 0.5) * self.cell_width
        lat, lon, _ = self.converter.NEU2LLA(n.item(), e.item(), self.altitude_meters.item())
        return lat, lon
    
    def get_rectangular_sensor_coverage_mask(self, 
                                             sensor_pos_lat: torch.Tensor, 
                                             sensor_pos_lon: torch.Tensor, 
                                             sensor_direction: torch.Tensor, #radians. 
                                             sensor_range: torch.Tensor, 
                                             sensor_width: torch.Tensor):
        #only supports sensors rotated by 90 degrees.
        masked_grid = torch.Tensor((*self.grid_patrol_task.batch_size, self.grid_patrol_task.H, self.grid_patrol_task.W), dtype = torch.bool)
        #convert to meters.
        neu = self.converter.LLA2NEU(sensor_pos_lat, sensor_pos_lon, self.anchor_alt)
        n, e, u = neu
        #
        #find all the covered grid cell points.

