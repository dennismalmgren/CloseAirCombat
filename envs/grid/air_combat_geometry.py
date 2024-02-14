import torch
import pymap3d
#Pymap3d should be ported to pytorch for speedup.

class LatLonNEUConverter:
    def __init__(self, device, 
                 lat_center: float = 60.0, 
                 lon_center: float = 120.0, 
                 alt_center: float = 0):
        self.device = device
        self.lat0 = lat_center
        self.lon0 = lon_center
        self.alt0 = alt_center

#add batching
    def LLA2NEU(self, lat, lon, alt):
        """Convert from Geodetic Coordinate System to NEU Coordinate System.

        Args:
            lat, lon, alt (float): target geodetic latitude(°), longitude(°), altitude(m)
            lat, lon, alt (float): observer geodetic latitude(°), longitude(°), altitude(m); Default=`(60°N, 120°E, 0m)`

        Returns:
            (torch.Tensor): (North, East, Up), unit: m
        """
        n, e, d = pymap3d.geodetic2ned(lat, lon, alt, self.lat0, self.lon0, self.alt0)
        return torch.tensor([n, e, -d], device = self.device)


    def NEU2LLA(self, n, e, u):
        """Convert from NEU Coordinate System to Geodetic Coordinate System.

        Args:
            n, e, u (float): target relative position w.r.t. North, East, Down
            lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(60°N, 120°E, 0m)`

        Returns:
            (torch.Tensor): (lat, lon, alt), unit: °, °, m
        """
        lat, lon, h = pymap3d.ned2geodetic(n, e, -u, self.lat0, self.lon0, self.alt0)
        return torch.tensor([lat, lon, h], device = self.device)


def get_AO_TA_R(ego_feature, enm_feature, return_side=False):
    """Get AO & TA angles and relative distance between two agent.

    Args:
        ego_feature & enemy_feature (tuple): (north, east, down, vn, ve, vd)

    Returns:
        (tuple): ego_AO, ego_TA, R
    """
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = torch.norm(torch.tensor([ego_vx, ego_vy, ego_vz]))
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = torch.norm(torch.tensor([enm_vx, enm_vy, enm_vz]))
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = torch.norm(torch.tensor([delta_x, delta_y, delta_z]))

    proj_dist = delta_x * ego_vx + delta_y * ego_vy + delta_z * ego_vz
    ego_AO = torch.acos(torch.clamp(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy + delta_z * enm_vz
    ego_TA = torch.acos(torch.clamp(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        #sign of the z-component of the cross product
        side_flag = torch.sign(ego_vx * delta_y - ego_vy * delta_x)
        return ego_AO, ego_TA, R, side_flag


def get2d_AO_TA_R(ego_feature, enm_feature, return_side=False):
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = torch.norm([ego_vx, ego_vy])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = torch.norm([enm_vx, enm_vy])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = torch.norm([delta_x, delta_y])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy
    ego_AO = torch.acos(torch.clamp(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy
    ego_TA = torch.acos(torch.clamp(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        #sign of the z-component of the cross product
        side_flag = torch.sign(ego_vx * delta_y - ego_vy * delta_x)
        return ego_AO, ego_TA, R, side_flag


def in_range_deg(angle):
    """ Given an angle in degrees, normalises in (-180, 180] """
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def in_range_rad(angle):
    """ Given an angle in rads, normalises in (-pi, pi] """
    angle = angle % (2 * torch.pi)
    if angle > torch.pi:
        angle -= 2 * torch.pi
    return angle
