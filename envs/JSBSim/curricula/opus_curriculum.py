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
        self.max_altitude_increment = 7000 #feet
        self.max_velocities_u_increment = 100 #m/s
        self.check_interval = 30 #seconds
        self.increment_size = [0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10

    def reset(self, task, env):
        pass
#        for agent_id, agent in env.agents.items():
            #get state.
    #   ic_long_gc_deg: 120.0,
    #   ic_lat_geod_deg: 60.0,
    #   ic_h_sl_ft: 20000,
    
        #agent = env.agents[agent_]
        # agent
        # env.agents[agent_id]set_property_value(self, prop, value):
        # for key, value in self.init_state.items():
        #     self.set_property_value(Catalog[key], value)
        #     'target_heading_deg': init_heading,
        #         'target_altitude_ft': init_altitude,
        #         'target_velocities_u_mps': init_velocities_u * 0.3048,
    
    def update_task(self, env, agent_id, info= {}):
        return
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft didn't reach the target heading in limited time.

        Args:
            task: task instance
            env: environment instance

        Returns:Q
            (tuple): (done, success, info)
        """
        done = False
        success = False
        cur_step = info['current_step']
        check_time = env.agents[agent_id].get_property_value(c.heading_check_time)
        #check time is initially 0. This task works because the agent was initialized with a delta heading of 0 (target heading == current heading)
        # check heading when simulation_time exceed check_time
        if env.agents[agent_id].get_property_value(c.simulation_sim_time_sec) >= check_time:
            if math.fabs(env.agents[agent_id].get_property_value(c.delta_heading)) > 10:
                done = True
            # if current target heading is reached, random generate a new target heading
            else:
                delta = self.increment_size[env.heading_turn_counts]
                delta_heading = env.np_random.uniform(-delta, delta) * self.max_heading_increment
                delta_altitude = env.np_random.uniform(-delta, delta) * self.max_altitude_increment
                delta_velocities_u = env.np_random.uniform(-delta, delta) * self.max_velocities_u_increment
                new_heading = env.agents[agent_id].get_property_value(c.target_heading_deg) + delta_heading
                new_heading = (new_heading + 360) % 360
                new_heading = new_heading * np.pi / 180
                new_altitude = env.agents[agent_id].get_property_value(c.travel_1_target_position_h_sl_m) + delta_altitude
                new_velocities_u = env.agents[agent_id].get_property_value(c.travel_1_target_velocities_u_mps) + delta_velocities_u

                env.agents[agent_id].set_property_value(c.travel_1_target_attitude_psi_rad, new_heading)
                env.agents[agent_id].set_property_value(c.travel_1_target_position_h_sl_m, new_altitude)
                env.agents[agent_id].set_property_value(c.travel_1_target_velocities_u_mps, new_velocities_u)
                env.agents[agent_id].set_property_value(c.heading_check_time, check_time + c.travel_1_target_time_s)
                env.heading_turn_counts += 1
                self.log(f'current_step:{cur_step} target_heading:{new_heading} '
                         f'target_altitude_m:{new_altitude} target_velocities_u_mps:{new_velocities_u}')
        if done:
            self.log(f'agent[{agent_id}] unreached heading. Total Steps={env.current_step}')
            info['heading_turn_counts'] = env.heading_turn_counts
        success = False
        return done, success, info
    