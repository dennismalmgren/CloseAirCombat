import math
from ..core.catalog import Catalog as c
from .curiculum_base import BaseCurriculum
import numpy as np
from ..utils.utils import LLA2NED, NED2LLA
from ..core.trajectory_planning import FlightModes, FlightProperties, Waypoint, FlightDirective

class OpusCurriculumMissile(BaseCurriculum):
    """
    Updates the current opus training task.
    """
    def __init__(self, config):
        super().__init__(config)



    def get_geo_boundaries(self, env):
        lat_min = self.env.task.lat0
        lon_min = self.env.task.lon0
        lat_max = 
    def get_init_state(self, agent_id):
        #hack. we know it's only one agent for now..
        return self.agent_init_states[agent_id]
    
    def create_init_states(self, env):
        agent_init_states = dict()
        for agent_id in env.agents:
            
            init_heading_deg = env.np_random.uniform(0., 180.)
            init_altitude_m = env.np_random.uniform(2500., 9000.)
            init_velocities_u_mps = env.np_random.uniform(120., 365.)

            agent_init_states[agent_id] = {
                'ic_psi_true_deg': init_heading_deg,
                'ic_h_sl_ft': init_altitude_m / 0.3048,
                'ic_u_fps': init_velocities_u_mps / 0.3048,
            }
        return agent_init_states
    
    def reset(self, env):
        for agent_id in env.agents:
            agent = env.agents[agent_id]
            self.create_missile(agent, env)
            
            
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
        agent = env.agents[agent_id]


    