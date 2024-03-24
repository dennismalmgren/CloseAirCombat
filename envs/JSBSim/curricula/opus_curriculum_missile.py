import math
from ..core.catalog import Catalog as c
from .curiculum_base import BaseCurriculum
import numpy as np
from ..utils.utils import LLA2NED, NED2LLA
from ..core.trajectory_planning import FlightModes, FlightProperties, Waypoint, FlightDirective
from ..core.simple_cruise_missile import randomize_starting_position, CruiseMissile, get_geopath
from ..core.catalog import Catalog as c

linkoping = (58.40160, 15.63308, 1000.0)
jonkoping = (57.75882, 14.16009, 1000.0)
norrkoping = (58.58341, 16.23918, 1000.0)
kaliningrad = (54.73032, 20.45923, 1000.0)
stpetersburg = (59.94279, 30.24516, 1000.0)
knackpunkt = (58.25883, 27.71385, 1000.0)

cm1 = ['cruise_missile_1',
       knackpunkt,
       [linkoping]]
cm2 = ['cruise_missile_2',
       knackpunkt,
       [linkoping]]
cm3 = ['cruise_missile_3',
       knackpunkt,
       [jonkoping]]
cm4 = ['cruise_missile_4',
       knackpunkt,
       [jonkoping]]
cm5 = ['cruise_missile_5',
       knackpunkt,
       [norrkoping]]
cm6 = ['cruise_missile_6',
       knackpunkt,
       [linkoping]]
cm7 = ['cruise_missile_7',
       knackpunkt,
       [norrkoping]]
cm8 = ['cruise_missile_8',
       knackpunkt,
       [linkoping]]
cm9 = ['cruise_missile_9',
       knackpunkt,
       [norrkoping]]
cm10 = ['cruise_missile_10',
        knackpunkt,
        [jonkoping]]

cms = [cm1, cm2, cm3, cm4, cm5, cm6, cm7, cm8, cm9, cm10]

class BoundingRect:
    def __init__(self, lat_min, lon_min, lat_max, lon_max):
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.lat_max = lat_max
        self.lon_max = lon_max

    def contains(self, lat, lon):
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max
    
class OpusCurriculumMissile(BaseCurriculum):
    """
    Updates the current opus training task.
    """
    def __init__(self, config):
        super().__init__(config)
        self.boundingRect = BoundingRect(56.98033, 14.07798, 58.74902, 20.82290)

    def get_init_state(self, agent_id):
        #hack. we know it's only one agent for now..
        return self.agent_init_states[agent_id]
    
    def create_init_states(self, env):
        agent_init_states = dict()
        for agent_id in env.agents:
            
            init_heading_deg = env.np_random.uniform(0., 180.)
            init_altitude_m = env.np_random.uniform(4500., 9000.)
            init_velocities_u_mps = env.np_random.uniform(120., 365.)

            agent_init_states[agent_id] = {
                'ic_psi_true_deg': init_heading_deg,
                'ic_h_sl_ft': init_altitude_m / 0.3048,
                'ic_u_fps': init_velocities_u_mps / 0.3048,
            }
        return agent_init_states
    
    def create_missile(self, agent, env):
        missile_index = env.np_random.integers(0, len(cms))
        missiles = [cms[missile_index]]
        randomize_starting_position(missiles)
        missile = missiles[0]
        self.cm = CruiseMissile(missile[0], missile[1], get_geopath(missile[2]))
        steps = env.np_random.integers(1350, 2300)
        for i in range(steps):
            self.cm.step(1.0)
        
        while not self.boundingRect.contains(self.cm.pose.position.latitude, self.cm.pose.position.longitude):
            self.cm.step(1.0)
        
        #okay, now we have a missile at random location, to intercept.
        
    def reset(self, env):
        for agent_id in env.agents:
            agent = env.agents[agent_id]
            self.create_missile(agent, env)
            dist = env.np_random() * 4000 + 1000 #meter
            altitude = 1000 #meter
            cm_ned_north, cm_ned_east, cm_ned_down = LLA2NED(self.cm.pose.position.latitude, self.cm.pose.position.longitude, self.cm.pose.position.altitude, env.task.lat0, env.task.lon0, env.task.alt0)
            ship_east = cm_ned_east + dist

            lat, lon, alt = NED2LLA(cm_ned_north, ship_east, cm_ned_down)
            env.task.set_property_value(c.position_h_sl_ft, alt / 0.3048)
            env.task.set_property_value(c.position_lat_gc_rad, lat * np.pi / 180) #
            env.task.set_property_value(c.position_lon_gc_rad, lon * np.pi / 180) #

    def step(self, env, agent_id, info= {}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft didn't reach the target heading in limited time.

        Args:
            task: task instance
            env: environment instance

        Returns:Q
            (tuple): (done, success, info)
        """
        f
        agent = env.agents[agent_id]


    