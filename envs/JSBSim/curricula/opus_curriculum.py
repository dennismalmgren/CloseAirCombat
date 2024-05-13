import math
from ..core.catalog import Catalog as c
from .curriculum_base import BaseCurriculum
import numpy as np
from ..tasks import OpusAltitudeTask, OpusAltitudeSpeedHeadingTask, OpusAltitudeSpeedTask

class OpusCurriculum(BaseCurriculum):
    """
    Updates the current opus training task.
    """
    def __init__(self, config):
        super().__init__(config)
        self.max_heading_increment = 180 #degrees
        self.max_altitude_increment = 2000 #m
        self.max_velocities_vc_mps_increment = 100 #m/s
        self.check_interval = 30 #seconds
        self.increment_size = [0.2, 0.4, 0.6, 0.8, 1.0] 
        self.heading_turn_counts = 0
        

    def get_init_state(self, agent_id):
        #hack. we know it's only one agent for now..
        return self.agent_init_states[agent_id]
    
    def create_init_states(self, env):
        agent_init_states = dict()
        for agent_id in env.agents:
            #we also need to update location.
            init_heading_deg = env.np_random.uniform(0., 180.)
            init_altitude_m = env.np_random.uniform(2500., 9000.)
            init_velocities_u_mps = env.np_random.uniform(120., 365.)
            init_lat_geod_deg = env.np_random.uniform(57.0, 60.0)
            init_long_gc_deg = env.np_random.uniform(15.0, 20.0)
            agent_init_states[agent_id] = {
                'ic_psi_true_deg': init_heading_deg,
                'ic_h_sl_ft': init_altitude_m / 0.3048,
                'ic_u_fps': init_velocities_u_mps / 0.3048,
                'ic_long_gc_deg': init_long_gc_deg,
                'ic_lat_geod_deg': init_lat_geod_deg,
            }
        return agent_init_states
    
    def reset(self, env):
        self.task.reset(env)

        self.heading_turn_counts = 0

        # for agent_id in env.agents:
        #     agent = env.agents[agent_id]
        #     current_heading_rad = agent.get_property_value(c.attitude_heading_true_rad) 
        #     current_speed = agent.get_property_value(c.velocities_vc_mps) 
        #     #current_time = agent.get_property_value(c.simulation_sim_time_sec) #will be at least.
        #     #also: set task values so that they can be returned by reset.
        #     #todo: move reset to the task.
        #     #agent.set_property_value(c.missions_cruise_target_attitude_heading_true_rad, current_heading_rad)
        #     #agent.set_property_value(c.missions_cruise_target_velocities_vc_mps, current_speed)
        #    # agent.set_property_value(c.travel_1_target_time_s, (self.check_interval + current_time))
        #     self.task.reset(env, agent_id)

    def load_task(self):
        #taskname = getattr(self.config, 'task', None)
        self.task = OpusAltitudeTask(self.config)
        return self.task
    
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
        current_time = agent.get_property_value(c.simulation_sim_time_sec)
        check_time = agent.get_property_value(c.travel_1_target_time_s)
        #check time is initially 0. This task works because the agent was initialized with a delta heading of 0 (target heading == current heading)
        # check heading when simulation_time exceed check_time

        if current_time >= check_time:
            heading_turn_count = min(self.heading_turn_counts, len(self.increment_size) - 1)
            delta = self.increment_size[heading_turn_count]
            delta_heading = env.np_random.uniform(-delta, delta) * self.max_heading_increment
            delta_altitude = env.np_random.uniform(-delta, delta) * self.max_altitude_increment
            delta_velocities_u = env.np_random.uniform(-delta, delta) * self.max_velocities_vc_mps_increment
            delta_time = env.np_random.uniform(30, 60)
            
            new_altitude = agent.get_property_value(c.missions_cruise_target_position_h_sl_m) + delta_altitude
            new_altitude = min(max(1000, new_altitude), 9000) #clamp to 500-9000m
            agent.set_property_value(c.missions_cruise_target_position_h_sl_m, new_altitude)

        #     #move from current, not the one we were aiming for.
        #     #not sure which property we compare with for this one.
            new_heading = agent.get_property_value(c.missions_cruise_target_attitude_heading_true_rad) * 180 / np.pi + delta_heading
            new_heading = (new_heading + 360) % 360
            new_heading = new_heading * np.pi / 180
            agent.set_property_value(c.missions_cruise_target_attitude_heading_true_rad, new_heading)

            new_velocities_u = agent.get_property_value(c.missions_cruise_target_velocities_vc_mps) + delta_velocities_u
            new_velocities_u = min(max(120, new_velocities_u), 365)
            agent.set_property_value(c.missions_cruise_target_velocities_vc_mps, new_velocities_u)
            
            new_time = delta_time + current_time
            agent.set_property_value(c.travel_1_target_time_s, new_time)
            
            self.heading_turn_counts += 1


    