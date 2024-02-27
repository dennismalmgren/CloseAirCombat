import math
from ..core.catalog import Catalog as c
from .curiculum_base import BaseCurriculum
import numpy as np

class OpusCurriculum(BaseCurriculum):
    """
    Updates the current opus training task.
    """
    def __init__(self, config):
        super().__init__(config)
        uid = list(config.aircraft_configs.keys())[0]
        aircraft_config = config.aircraft_configs[uid]
        self.max_heading_increment = 180 #degrees
        self.max_altitude_increment = 2000 #m
        self.max_velocities_u_increment = 100 #m/s
        self.check_interval = 30 #seconds
        self.increment_size = [0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 100
        self.heading_turn_counts = 0

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
        self.heading_turn_counts = 0
        for agent_id in env.agents:
            agent = env.agents[agent_id]
            current_altitude = agent.get_property_value(c.position_h_sl_m)
            current_heading_rad = agent.get_property_value(c.attitude_heading_true_rad) 
            current_speed = agent.get_property_value(c.velocities_u_mps) 
            current_time = agent.get_property_value(c.simulation_sim_time_sec) #will be at least.
            #also: set task values so that they can be returned by reset.
            agent.set_property_value(c.current_task_id, 0) #0 or 1. we always go with 0 for now..
            agent.set_property_value(c.task_1_type_id, 1) #1 for heading, 2 for waypoint.
            agent.set_property_value(c.task_2_type_id, 0) #0 for no mission.
            agent.set_property_value(c.travel_1_target_position_h_sl_m, current_altitude)
            agent.set_property_value(c.travel_1_target_attitude_psi_rad, current_heading_rad)
            agent.set_property_value(c.travel_1_target_velocities_u_mps, current_speed)
            agent.set_property_value(c.travel_1_target_time_s, (self.check_interval + current_time))
            
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
            delta = self.increment_size[self.heading_turn_counts]
            delta_heading = env.np_random.uniform(-delta, delta) * self.max_heading_increment
            delta_altitude = env.np_random.uniform(-delta, delta) * self.max_altitude_increment
            delta_velocities_u = env.np_random.uniform(-delta, delta) * self.max_velocities_u_increment
            delta_time = env.np_random.uniform(10, 30)
            
            new_altitude = agent.get_property_value(c.travel_1_target_position_h_sl_m) + delta_altitude
            new_altitude = min(max(600, new_altitude), 10000) #clamp to 500-10000m
            agent.set_property_value(c.travel_1_target_position_h_sl_m, new_altitude)

            #move from current, not the one we were aiming for.
            #not sure which property we compare with for this one.
            new_heading = agent.get_property_value(c.travel_1_target_attitude_psi_rad) * 180 / np.pi + delta_heading
            new_heading = (new_heading + 360) % 360
            new_heading = new_heading * np.pi / 180
            agent.set_property_value(c.travel_1_target_attitude_psi_rad, new_heading)

            new_velocities_u = agent.get_property_value(c.travel_1_target_velocities_u_mps) + delta_velocities_u
            agent.set_property_value(c.travel_1_target_velocities_u_mps, new_velocities_u)
            
            new_time = delta_time + current_time
            agent.set_property_value(c.travel_1_target_time_s, new_time)
            
            self.heading_turn_counts += 1


    